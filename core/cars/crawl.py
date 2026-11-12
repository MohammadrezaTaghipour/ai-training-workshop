"""Crawl Divar car listings via the site JSON API (lazy-load pagination).

Session 11 named Divar + Digikala-style APIs. Divar search pages load more
posts with ``POST /v8/postlist/w/search``; detail specs come from
``GET /v8/posts-v2/web/{token}``.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

import requests

DEFAULT_URL = "https://divar.ir/s/tehran/car/peugeot/207i"
MAX_LISTINGS = 50
_SEARCH_API = "https://api.divar.ir/v8/postlist/w/search"
_DETAIL_API = "https://api.divar.ir/v8/posts-v2/web/{token}"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Origin": "https://divar.ir",
}


class CrawlError(RuntimeError):
    """Divar crawl failed."""


ProgressCb = Callable[[str], None]


def crawl_listings(
    url: str = DEFAULT_URL,
    *,
    max_listings: int = MAX_LISTINGS,
    fetch_details: bool = True,
    pause_s: float = 0.25,
    progress: ProgressCb | None = None,
) -> list[dict[str, Any]]:
    """Collect up to ``max_listings`` car ads from a Divar search URL."""
    if max_listings < 1:
        raise CrawlError("max_listings must be at least 1.")

    session = requests.Session()
    session.headers.update(_HEADERS)
    session.headers["Referer"] = url

    _emit(progress, "Loading search page…")
    pre = _load_preloaded(session, url)
    search_data = _search_data_from_preloaded(pre)
    city_ids = _city_ids_from_preloaded(pre)
    cards = _cards_from_preloaded(pre)
    pagination = ((pre.get("nb") or {}).get("pagination") or {})
    pag_data = pagination.get("data")
    has_more = _has_more(pagination)

    _emit(progress, f"Page 1: {len(cards)} listings from page HTML.")

    seen: set[str] = set()
    cards = _unique_cards(cards, seen)

    page = 1
    while len(cards) < max_listings and has_more and pag_data:
        page += 1
        _emit(progress, f"Fetching page {page} via Divar API…")
        payload = _search_body(search_data, city_ids, pag_data)
        response = session.post(_SEARCH_API, json=payload, timeout=30)
        if response.status_code != 200:
            raise CrawlError(
                f"Search API failed ({response.status_code}): {response.text[:200]}"
            )
        data = response.json()
        batch = _cards_from_api_widgets(data.get("list_widgets") or [])
        before = len(cards)
        cards.extend(_unique_cards(batch, seen))
        added = len(cards) - before
        _emit(progress, f"Page {page}: +{added} listings (total {len(cards)}).")
        if added == 0:
            break
        pagination = data.get("pagination") or {}
        pag_data = pagination.get("data")
        has_more = _has_more(pagination)

    cards = cards[:max_listings]
    if not cards:
        raise CrawlError("No listings found for this URL.")

    if not fetch_details:
        return cards

    rows: list[dict[str, Any]] = []
    for i, card in enumerate(cards, start=1):
        token = card.get("token")
        _emit(progress, f"Detail {i}/{len(cards)}: {token}")
        detail = _fetch_detail(session, token) if token else {}
        rows.append({**card, **detail})
        if pause_s > 0 and i < len(cards):
            time.sleep(pause_s)
    return rows


def _emit(progress: ProgressCb | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _load_preloaded(session: requests.Session, url: str) -> dict[str, Any]:
    html_headers = {
        **_HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    response = session.get(url, headers=html_headers, timeout=30)
    if response.status_code != 200:
        raise CrawlError(f"Could not open URL ({response.status_code}).")
    match = re.search(r"window\.__PRELOADED_STATE__\s*=\s*", response.text)
    if not match:
        raise CrawlError("Divar preloaded state not found (page layout may have changed).")
    start = match.end()
    end = response.text.find("</script>", start)
    if end < 0:
        raise CrawlError("Could not parse Divar preloaded state.")
    blob = response.text[start:end].strip().rstrip(";")
    try:
        return json.loads(blob)
    except json.JSONDecodeError as exc:
        raise CrawlError("Divar preloaded state was not valid JSON.") from exc


def _search_data_from_preloaded(pre: dict[str, Any]) -> dict[str, Any]:
    raw = ((pre.get("search") or {}).get("searchData")) or {}
    form = raw.get("formData") or raw.get("form_data") or {}
    server = raw.get("serverPayload") or raw.get("server_payload") or {}
    if not form:
        raise CrawlError("Search filters missing from Divar page.")
    return {
        "form_data": {"data": form},
        "server_payload": server
        or {
            "@type": "type.googleapis.com/widgets.SearchData.ServerPayload",
            "additional_form_data": {
                "data": {"sort": {"str": {"value": "sort_date"}}}
            },
        },
    }


def _city_ids_from_preloaded(pre: dict[str, Any]) -> list[str]:
    text = json.dumps(pre.get("nb") or {}, ensure_ascii=False)
    # Tehran search pages use place_id / city id "1".
    ids = re.findall(r'"place_id"\s*:\s*"(\d+)"', text)
    if ids:
        return [ids[0]]
    return ["1"]


def _cards_from_preloaded(pre: dict[str, Any]) -> list[dict[str, Any]]:
    widgets = ((pre.get("nb") or {}).get("listWidgets")) or []
    cards: list[dict[str, Any]] = []
    for widget in widgets:
        data = widget.get("data") or {}
        dto = data.get("dto") or {}
        row = dto.get("data") if isinstance(dto, dict) else None
        if not isinstance(row, dict):
            row = data if data.get("token") else None
        if not isinstance(row, dict):
            continue
        card = _card_from_post_row(row)
        if card:
            cards.append(card)
    return cards


def _cards_from_api_widgets(widgets: list[Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for widget in widgets:
        if not isinstance(widget, dict):
            continue
        row = widget.get("data") or {}
        card = _card_from_post_row(row)
        if card:
            cards.append(card)
    return cards


def _card_from_post_row(row: dict[str, Any]) -> dict[str, Any] | None:
    token = row.get("token")
    if not token:
        action = row.get("action") or {}
        payload = action.get("payload") or {}
        token = payload.get("token")
    if not token:
        return None
    return {
        "token": token,
        "title": row.get("title") or "",
        "list_mileage_text": row.get("top_description_text")
        or row.get("topDescriptionText")
        or "",
        "list_price_text": row.get("middle_description_text")
        or row.get("middleDescriptionText")
        or "",
        "district_text": row.get("bottom_description_text")
        or row.get("bottomDescriptionText")
        or "",
        "url": f"https://divar.ir/v/{token}",
    }


def _unique_cards(
    cards: list[dict[str, Any]], seen: set[str]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for card in cards:
        token = card.get("token")
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(card)
    return out


def _has_more(pagination: dict[str, Any]) -> bool:
    if "hasMore" in pagination:
        return bool(pagination.get("hasMore"))
    if "has_next_page" in pagination:
        return bool(pagination.get("has_next_page"))
    return False


def _search_body(
    search_data: dict[str, Any],
    city_ids: list[str],
    pagination_data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "request_trigger": "REQUEST_TRIGGER_NORMAL",
        "pagination_data": pagination_data,
        "disable_recommendation": False,
        "map_state": {"camera_info": {"bbox": {}}},
        "search_data": search_data,
        "city_ids": [str(c) for c in city_ids],
        "user_selected_location": {
            "places": [{"place_id": str(city_ids[0])}]
        },
        "previous_user_selected_location": {"places": []},
    }


def _fetch_detail(session: requests.Session, token: str) -> dict[str, Any]:
    response = session.get(_DETAIL_API.format(token=token), timeout=30)
    if response.status_code != 200:
        return {"detail_error": f"HTTP {response.status_code}"}
    payload = response.json()
    info = _parse_detail(payload)
    info["brand_model"] = ((payload.get("webengage") or {}).get("brand_model")) or ""
    return info


def _parse_detail(payload: dict[str, Any]) -> dict[str, Any]:
    kv: dict[str, str] = {}
    scores: dict[str, str] = {}
    for section in payload.get("sections") or []:
        for widget in section.get("widgets") or []:
            if not isinstance(widget, dict):
                continue
            wtype = widget.get("widget_type") or ""
            data = widget.get("data") or {}
            if wtype == "GROUP_INFO_ROW":
                for item in data.get("items") or []:
                    title = str(item.get("title") or "").strip()
                    value = str(item.get("value") or "").strip()
                    if title:
                        kv[title] = value
            elif wtype == "UNEXPANDABLE_ROW":
                title = str(data.get("title") or "").strip()
                value = str(data.get("value") or "").strip()
                if title:
                    kv[title] = value
            elif wtype == "SCORE_ROW":
                title = str(data.get("title") or "").strip()
                value = str(data.get("descriptive_score") or "").strip()
                if title:
                    scores[title] = value

    web = payload.get("webengage") or {}
    price = web.get("price")
    return {
        "mileage_text": kv.get("کارکرد", ""),
        "year_text": kv.get("مدل (سال تولید)", ""),
        "color": kv.get("رنگ", ""),
        "transmission": kv.get("گیربکس", ""),
        "fuel": kv.get("نوع سوخت", ""),
        "price_text": kv.get("قیمت پایه", ""),
        "price": price if isinstance(price, (int, float)) else None,
        "engine_status": scores.get("موتور", ""),
        "chassis_status": scores.get("وضعیت شاسی‌ها", ""),
        "body_status": scores.get("بدنه", ""),
        "gearbox_status": scores.get("گیربکس", ""),
    }
