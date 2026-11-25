"""Dialogue schema shared by the data pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class Turn:
    speaker: str           # "agent" | "user"
    text: str
    backchannel: bool = False   # short user utterance spoken OVER the agent
    interrupt: bool = False     # user barges in, cutting the agent off
    pause_mid: bool = False     # user pauses 0.5-1.5 s mid-turn (agent must wait)

    def validate(self):
        assert self.speaker in ("agent", "user"), self.speaker
        assert self.text.strip(), "empty turn text"
        if self.backchannel or self.interrupt:
            assert self.speaker == "user", "only the user can backchannel/interrupt"


@dataclass
class Dialogue:
    system_prompt: str
    turns: list[Turn] = field(default_factory=list)
    agent_voice: str = "af_heart"
    user_voice: str = "am_adam"

    def validate(self):
        assert self.system_prompt.strip(), "empty system prompt"
        assert len(self.turns) >= 4, f"too few turns: {len(self.turns)}"
        has_agent = any(t.speaker == "agent" and not t.backchannel for t in self.turns)
        assert has_agent, "dialogue has no agent turns"
        for t in self.turns:
            t.validate()
        # A backchannel/interrupt needs a preceding agent turn to overlap.
        for i, t in enumerate(self.turns):
            if t.backchannel or t.interrupt:
                assert any(x.speaker == "agent" and not x.backchannel and not x.interrupt
                           for x in self.turns[:i]), f"turn {i} has no agent turn to overlap"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "Dialogue":
        turns = [Turn(
            speaker=t["speaker"], text=t["text"],
            backchannel=bool(t.get("backchannel", False)),
            interrupt=bool(t.get("interrupt", False)),
            pause_mid=bool(t.get("pause_mid", False)),
        ) for t in d["turns"]]
        dlg = cls(
            system_prompt=d["system_prompt"], turns=turns,
            agent_voice=d.get("agent_voice", "af_heart"),
            user_voice=d.get("user_voice", "am_adam"),
        )
        dlg.validate()
        return dlg
