"""Tests for core.personaplex.voices -- Voice Prompt Library.

Covers:
- VoiceGender and VoiceStyle enums
- VoicePrompt data class
- VOICE_LIBRARY registry
- Lookup functions: get_voice, get_voices_by_gender, etc.
"""

from pathlib import Path

import pytest

from core.personaplex.voices import (
    VOICE_LIBRARY,
    VoiceGender,
    VoicePrompt,
    VoiceStyle,
    get_recommended_voice,
    get_voice,
    get_voices_by_gender,
    get_voices_by_style,
    resolve_voice_path,
)


class TestVoiceEnums:

    def test_gender_values(self):
        assert VoiceGender.FEMALE.value == "female"
        assert VoiceGender.MALE.value == "male"

    def test_style_values(self):
        expected = {
            "warm", "clear", "authoritative", "friendly",
            "scholarly", "professional", "calm", "commanding",
            "expressive", "varied",
        }
        actual = {s.value for s in VoiceStyle}
        assert actual == expected


class TestVoicePrompt:

    def test_path_property(self):
        vp = VoicePrompt(
            code="TEST",
            filename="TEST.pt",
            gender=VoiceGender.FEMALE,
            style=VoiceStyle.WARM,
            description="Test voice",
            recommended_for=["testing"],
        )
        assert vp.path == "TEST.pt"


class TestVoiceLibrary:

    def test_library_has_entries(self):
        assert len(VOICE_LIBRARY) > 0

    def test_all_entries_have_required_fields(self):
        for code, vp in VOICE_LIBRARY.items():
            assert vp.code == code
            assert vp.filename.endswith(".pt")
            assert isinstance(vp.gender, VoiceGender)
            assert isinstance(vp.style, VoiceStyle)
            assert len(vp.description) > 0
            assert len(vp.recommended_for) > 0

    def test_nat_and_var_categories(self):
        nat_voices = [c for c in VOICE_LIBRARY if c.startswith("NAT")]
        var_voices = [c for c in VOICE_LIBRARY if c.startswith("VAR")]
        assert len(nat_voices) >= 8  # 4 female + 4 male natural voices
        assert len(var_voices) >= 5  # At least some variety voices

    def test_unique_filenames(self):
        filenames = [vp.filename for vp in VOICE_LIBRARY.values()]
        assert len(filenames) == len(set(filenames))


class TestGetVoice:

    def test_existing_voice(self):
        v = get_voice("NATM1")
        assert v is not None
        assert v.code == "NATM1"

    def test_case_insensitive(self):
        v = get_voice("natm1")
        assert v is not None

    def test_nonexistent_voice(self):
        v = get_voice("NONEXISTENT")
        assert v is None


class TestGetVoicesByGender:

    def test_female_voices(self):
        females = get_voices_by_gender(VoiceGender.FEMALE)
        assert len(females) > 0
        assert all(v.gender == VoiceGender.FEMALE for v in females)

    def test_male_voices(self):
        males = get_voices_by_gender(VoiceGender.MALE)
        assert len(males) > 0
        assert all(v.gender == VoiceGender.MALE for v in males)


class TestGetVoicesByStyle:

    def test_professional_style(self):
        voices = get_voices_by_style(VoiceStyle.PROFESSIONAL)
        assert len(voices) >= 1

    def test_varied_style(self):
        voices = get_voices_by_style(VoiceStyle.VARIED)
        assert len(voices) >= 5  # Multiple variety voices


class TestGetRecommendedVoice:

    def test_known_use_case(self):
        v = get_recommended_voice("security")
        assert v is not None

    def test_unknown_use_case(self):
        v = get_recommended_voice("nonexistent_use_case_xyz")
        assert v is None

    def test_reasoning_use_case(self):
        v = get_recommended_voice("reasoning")
        assert v is not None
        assert v.style == VoiceStyle.CLEAR


class TestResolveVoicePath:

    def test_known_voice(self):
        path = resolve_voice_path("NATM1")
        assert isinstance(path, Path)
        assert str(path).endswith("NATM1.pt")

    def test_custom_base_dir(self):
        path = resolve_voice_path("NATM1", base_dir=Path("/tmp/voices"))
        assert str(path) == "/tmp/voices/NATM1.pt"

    def test_unknown_voice_raises(self):
        with pytest.raises(ValueError, match="Unknown voice code"):
            resolve_voice_path("NONEXISTENT")
