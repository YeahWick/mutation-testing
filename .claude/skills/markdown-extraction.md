# Markdown Section Extraction Skill

Use this skill when you need to extract specific sections from markdown documentation files.

## When to Use This Skill

- User asks about specific sections of project documentation
- Need to retrieve targeted information from markdown files
- Want to list available documentation sections
- Need to extract all sections at a specific heading level

## Available Commands

The `md-extract.sh` script provides three main commands:

### 1. List Headers
Lists all headers in a markdown file with their line numbers and levels.

```bash
./md-extract.sh <file.md> list
```

**Example:**
```bash
./md-extract.sh claude.md list
```

**Output format:**
```
123  L1: Project Overview
145  L2: Core Goals
167  L3: Runtime Mutation Injection
```

### 2. Extract Section by Header Name
Extracts a specific section and all its subsections by header name.

```bash
./md-extract.sh <file.md> extract "<header-name>"
```

**Examples:**
```bash
# Extract the "Core Goals" section
./md-extract.sh claude.md extract "Core Goals"

# Extract the "Architecture" section
./md-extract.sh docs/pytest_mutation_plan.md extract "Architecture"

# Extract nested section
./md-extract.sh claude.md extract "Runtime Mutation Injection"
```

**Behavior:**
- Returns the header and all content until the next header of same or higher level
- Includes all subsections within the extracted section
- Case-sensitive header matching

### 3. Extract All Sections at Level
Extracts all sections at a specific heading level (1-6).

```bash
./md-extract.sh <file.md> extract-level <1-6>
```

**Examples:**
```bash
# Extract all top-level sections (# headers)
./md-extract.sh claude.md extract-level 1

# Extract all second-level sections (## headers)
./md-extract.sh claude.md extract-level 2
```

## Available Documentation Files

- `claude.md` - Project overview and goals
- `docs/pytest_mutation_plan.md` - Detailed implementation plan
- `doc/ast_parsing_guide.md` - AST parsing reference
- `README.md` - Basic project description

## Usage Patterns

### Pattern 1: User asks about project goals
```
User: "What are the goals of this project?"

Agent: Let me extract the goals section from the project documentation.
[Run: ./md-extract.sh claude.md extract "Core Goals"]
[Present the extracted content to user]
```

### Pattern 2: User wants overview of documentation structure
```
User: "What documentation is available?"

Agent: Let me list the sections in the main documentation.
[Run: ./md-extract.sh claude.md list]
[Present formatted list to user]
```

### Pattern 3: User asks about specific technical topic
```
User: "How does the architecture work?"

Agent: Let me extract the architecture section.
[Run: ./md-extract.sh claude.md extract "Architecture"]
[Present the content and answer follow-up questions]
```

### Pattern 4: User wants all main sections
```
User: "Give me an overview of all main topics"

Agent: Let me extract all top-level sections.
[Run: ./md-extract.sh claude.md extract-level 1]
[Summarize the sections for the user]
```

## Implementation Guidelines

1. **Always verify file exists** before running extraction
2. **Quote header names** when they contain spaces
3. **Use absolute or relative paths** correctly based on current directory
4. **Present extracted content** in a readable format to the user
5. **Offer to extract related sections** if user needs more context

## Error Handling

If a header is not found:
```
Error: Header 'X' not found in file.md
```

Solution: Run `list` command first to see available headers with correct spelling.

If file is not found:
```
Error: File 'X' not found
```

Solution: Check the file path and current working directory.

## Example Session

```
User: "What's this project about?"

Agent: Let me get the project overview for you.

[Runs: ./md-extract.sh claude.md extract "Project Overview"]

Agent: This is a Python mutation testing framework that validates test suite
quality by injecting runtime mutations and verifying tests catch them...
[continues with extracted content]

Would you like me to extract more details about any specific aspect?
```

## Tips for Best Results

- Start with `list` to see structure if unfamiliar with the file
- Use `extract` for focused information retrieval
- Use `extract-level 2` to get an overview of major topics
- Chain extractions: first get overview, then drill into specific sections
- Always provide context from the extracted content in your response

## Script Location

The script is located at: `./md-extract.sh` (project root)

Make sure you're in the project root directory when running commands, or use the full path.
