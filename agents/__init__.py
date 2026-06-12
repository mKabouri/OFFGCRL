from agents.hiql import HIQLAgent
from agents.crl import CRLAgent
from agents.flow.fql import FQLAgent
from agents.tdinfonce import TDINFONCEAgent

agents = dict(
    hiql=HIQLAgent,
    crl=CRLAgent,
    fql=FQLAgent,
    tdinfonce=TDINFONCEAgent,
)
