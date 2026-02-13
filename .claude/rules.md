# Code Quality Rules

## Configuration and Templates

- Never hardcode prompts, instructions, or templates in code files
- Store configuration data in separate files (YAML, XML, JSON, etc.)
- Use precise, descriptive field names (e.g., `system_prompt`, `instruction`) instead of generic ones (e.g., `template`)
- Code should load and render templates, not define them

## Code Maintenance

- Remove backward compatibility shims and deprecated code immediately after migration
- Don't keep redundant aliases or compatibility layers "just in case"
- Prefer clean, current code over maintaining legacy patterns
