"""Shared Direction A event ontology."""

EVENT_TYPES = {
    "M1": "Interest Rate Change",
    "M2": "Economic Data Release",
    "M3": "Trade Policy Change",
    "M4": "Monetary Policy Signal",
    "M5": "Inflation Data",
    "M6": "Regulatory Change",
    "C1": "Earnings Report",
    "C2": "Merger & Acquisition",
    "C3": "Executive Change",
    "C4": "Product Launch",
    "C5": "Stock Buyback",
    "C6": "Stock Split",
    "C7": "Litigation / Fine",
    "C8": "Bankruptcy / Default",
    "K1": "Analyst Rating Change",
    "K2": "Insider Trading",
    "K3": "Index Rebalancing",
    "K4": "Technical Breakout",
    "G1": "Geopolitical Conflict",
    "G2": "Disaster / Health Crisis",
}

NUM_EVENT_TYPES = len(EVENT_TYPES)
EVENT_TYPE_LIST = list(EVENT_TYPES.keys())
EVENT_TYPE_TO_ID = {event_type: idx for idx, event_type in enumerate(EVENT_TYPE_LIST)}
ID_TO_EVENT_TYPE = {idx: event_type for event_type, idx in EVENT_TYPE_TO_ID.items()}
