"""Contracts for the skill activation guidance shown to the agent."""

import pytest

from ptc_agent.agent.middleware.skills.middleware import SkillsMiddleware
from ptc_agent.agent.middleware.skills.registry import SkillDefinition


def test_ptc_manifest_defines_read_as_the_complete_activation_step() -> None:
    middleware = SkillsMiddleware(mode="ptc")

    manifest = middleware.build_manifest({})

    assert manifest is not None
    assert "Read tool" in manifest
    assert "complete activation step" in manifest
    assert "direct tool calls" in manifest
    assert "Do NOT use Bash, Glob, or Grep" in manifest
    assert "To activate, read" not in manifest


def test_flash_manifest_keeps_load_skill_guidance() -> None:
    middleware = SkillsMiddleware(mode="flash")

    manifest = middleware.build_manifest({})

    assert manifest is not None
    assert "Call `LoadSkill` with the skill name" in manifest


@pytest.mark.asyncio
async def test_ptc_skill_result_guides_direct_tool_use() -> None:
    skill = SkillDefinition(
        name="demo",
        description="Demo skill",
        tools=[],
        skill_md_path="skills/demo/SKILL.md",
    )
    middleware = SkillsMiddleware(mode="ptc", skill_registry={"demo": skill})

    result = await middleware._build_skill_result(skill)

    assert "skill is active" in result
    assert "Use the listed tools as direct tool calls" in result
    assert "before using the skill tools" not in result
