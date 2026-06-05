from agents.hiql import HIQLAgent
from agents.crl import CRLAgent
from agents.flow.fql import FQLAgent

agents = dict(
    hiql=HIQLAgent,
    crl=CRLAgent,
    fql=FQLAgent,
)
