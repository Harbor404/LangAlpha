"""Contracts for the skill activation guidance shown to the agent."""

import pytest
from langchain_core.messages import ToolMessage

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


class _ReadRequest:
    """Minimal stand-in for the middleware request object."""

    def __init__(self, file_path: str) -> None:
        self.tool_call = {"name": "Read", "args": {"file_path": file_path}}


def _ptc_middleware() -> SkillsMiddleware:
    skill = SkillDefinition(
        name="demo",
        description="Demo skill",
        tools=[],
        skill_md_path="skills/demo/SKILL.md",
    )
    return SkillsMiddleware(mode="ptc", skill_registry={"demo": skill})


@pytest.mark.asyncio
async def test_ptc_read_activation_announces_the_skill_in_the_read_result() -> None:
    """The Read result is what the agent sees at activation, so it must say so."""

    async def handler(_request):
        return ToolMessage(content="# Demo\nbody", tool_call_id="call-1", name="Read")

    command = await _ptc_middleware().awrap_tool_call(
        _ReadRequest("skills/demo/SKILL.md"),
        handler,
    )

    assert command.update["loaded_skills"] == ["demo"]
    message = command.update["messages"][0]
    # The file content is preserved; the activation note is appended.
    assert message.content.startswith("# Demo\nbody")
    assert "[Skill activated: demo]" in message.content
    assert "direct tool calls" in message.content


@pytest.mark.asyncio
async def test_ptc_read_of_an_unregistered_skill_md_is_untouched() -> None:
    async def handler(_request):
        return ToolMessage(content="# Other", tool_call_id="call-2", name="Read")

    result = await _ptc_middleware().awrap_tool_call(
        _ReadRequest("skills/other/SKILL.md"),
        handler,
    )

    assert result.content == "# Other"


@pytest.mark.asyncio
async def test_ptc_read_failure_does_not_activate_the_skill() -> None:
    """A failed read never delivered the docs, so the skill must stay inactive."""

    async def handler(_request):
        return ToolMessage(
            content="Error: file not found",
            tool_call_id="call-3",
            name="Read",
            status="error",
        )

    result = await _ptc_middleware().awrap_tool_call(
        _ReadRequest("skills/demo/SKILL.md"),
        handler,
    )

    assert result.content == "Error: file not found"
    assert "Skill activated" not in result.content
