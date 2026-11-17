"""Ticket taxonomy and the ITIL impact x urgency priority matrix.

Priority is *derived*, never guessed: each synthetic ticket is generated from an explicit
(impact, urgency) pair and its label comes from the standard ITIL matrix below.
"""

from __future__ import annotations

CATEGORIES: dict[str, list[str]] = {
    "access": [
        "account locked after too many sign-in attempts",
        "password expired and the reset portal rejects the new password",
        "lost or replaced phone, so the MFA authenticator no longer works",
        "needs access to a shared folder or SharePoint site",
        "new employee's account cannot sign in on day one",
        "missing permissions in a business application after a role change",
        "single sign-on loops back to the login page",
    ],
    "network": [
        "Wi-Fi keeps disconnecting in one area of the building",
        "VPN client cannot connect from home",
        "internet is extremely slow",
        "wired network port at the desk has no connection",
        "internal website does not resolve or times out",
        "mapped network drive is missing or unreachable",
        "network switch or access point appears to be down",
    ],
    "email": [
        "Outlook crashes or freezes on start",
        "mailbox is full and cannot receive mail",
        "emails with attachments are not delivered",
        "shared mailbox does not appear in Outlook",
        "calendar invitations are not syncing to the phone",
        "messages from an important external sender go to junk",
    ],
    "hardware": [
        "laptop does not power on",
        "cracked or flickering laptop screen",
        "docking station does not detect external monitors",
        "keyboard or mouse stopped working",
        "laptop battery is swollen or drains very fast",
        "barcode scanner or label terminal on the shop floor is broken",
        "request for a replacement or new laptop",
    ],
    "software": [
        "request to install an application that is not in Software Center",
        "business application crashes with an error message",
        "software license expired",
        "Windows or application update failed and keeps retrying",
        "Excel macro or add-in throws an error",
        "ERP / MES system shows an error when posting transactions",
        "Teams audio or video does not work in meetings",
    ],
    "printing": [
        "printer is jammed or shows an error",
        "print jobs never come out at the printer",
        "badge release at the printer does not work",
        "printer is out of toner",
        "scan-to-email on the multifunction printer fails",
        "label printer on the production line prints garbage",
    ],
    "security": [
        "suspicious phishing email asking for credentials",
        "clicked a link and typed the password on a fake page",
        "pop-up claiming the computer is infected",
        "laptop or phone with company data was lost or stolen",
        "unexpected MFA prompts or sign-in alerts not started by the user",
        "unknown USB stick found in the parking lot or office",
        "colleague's account is sending strange emails",
    ],
}

IMPACT = {
    "low": "only the person writing the ticket is affected",
    "medium": "a whole team or department (around 5-30 people) is affected",
    "high": "an entire site, a production line, or customers are affected",
}
URGENCY = {
    "low": "it is an inconvenience and can wait a few days; there is a workaround",
    "medium": "it blocks normal work today; there is no good workaround",
    "high": "it must be fixed right now: work has stopped, money or safety or a security risk "
    "is involved",
}

# ITIL matrix: (impact, urgency) -> priority. P1 = critical ... P4 = low.
PRIORITY_MATRIX: dict[tuple[str, str], str] = {
    ("high", "high"): "P1",
    ("high", "medium"): "P2",
    ("medium", "high"): "P2",
    ("high", "low"): "P3",
    ("medium", "medium"): "P3",
    ("low", "high"): "P3",
    ("medium", "low"): "P4",
    ("low", "medium"): "P4",
    ("low", "low"): "P4",
}
PRIORITIES = ["P1", "P2", "P3", "P4"]
PRIORITY_NAMES = {"P1": "critical", "P2": "high", "P3": "medium", "P4": "low"}
CATEGORY_LIST = list(CATEGORIES)


def priority(impact: str, urgency: str) -> str:
    return PRIORITY_MATRIX[(impact, urgency)]
