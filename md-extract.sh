#!/bin/bash
# md-extract.sh - Extract sections from markdown files based on headers
#
# Usage:
#   ./md-extract.sh <file> list                    - List all headers
#   ./md-extract.sh <file> extract <header>        - Extract section by header
#   ./md-extract.sh <file> extract-level <level>   - Extract all headers of level (e.g., 1, 2)

set -euo pipefail

usage() {
    cat << EOF
Usage: $0 <markdown-file> <command> [args]

Commands:
    list                      List all headers with their levels
    extract <header>          Extract section content by header name
    extract-level <level>     Extract all sections at header level (1-6)
    help                      Show this help message

Examples:
    $0 claude.md list
    $0 claude.md extract "Core Goals"
    $0 claude.md extract-level 2
    $0 docs/plan.md extract "Architecture"
EOF
    exit 1
}

# Check arguments
if [ $# -lt 2 ]; then
    usage
fi

FILE="$1"
COMMAND="$2"

# Validate file exists
if [ ! -f "$FILE" ]; then
    echo "Error: File '$FILE' not found" >&2
    exit 1
fi

# List all headers
list_headers() {
    local file="$1"
    grep -n '^#' "$file" | while IFS=: read -r line_num content; do
        # Count hash marks to determine level
        level=$(echo "$content" | sed -E 's/^(#+).*/\1/' | tr -d '\n' | wc -c)
        # Extract header text (remove hashes and trim)
        text=$(echo "$content" | sed -E 's/^#+\s*//' | sed -E 's/\s*$//')
        printf "%-4s L%d: %s\n" "$line_num" "$level" "$text"
    done
}

# Extract section by header name
extract_section() {
    local file="$1"
    local header="$2"

    # Escape special regex characters in header
    local escaped_header=$(printf '%s\n' "$header" | sed 's/[]\/$*.^[]/\\&/g')

    # Find the line number of the header
    local start_line=$(grep -n "^#\+\s\+${escaped_header}\s*$" "$file" | head -1 | cut -d: -f1)

    if [ -z "$start_line" ]; then
        echo "Error: Header '$header' not found in $file" >&2
        return 1
    fi

    # Get the level of this header
    local header_content=$(sed -n "${start_line}p" "$file")
    local level=$(echo "$header_content" | sed -E 's/^(#+).*/\1/' | tr -d '\n' | wc -c)

    # Find the next header of same or higher level (lower number)
    local end_line=$(awk -v start="$start_line" -v lvl="$level" '
        NR > start && /^#/ {
            # Count hashes
            match($0, /^#+/)
            curr_level = RLENGTH
            if (curr_level <= lvl) {
                print NR - 1
                exit
            }
        }
    ' "$file")

    # If no end found, go to end of file
    if [ -z "$end_line" ]; then
        end_line=$(wc -l < "$file")
    fi

    # Extract the section
    sed -n "${start_line},${end_line}p" "$file"
}

# Extract all headers of a specific level
extract_by_level() {
    local file="$1"
    local target_level="$2"

    # Validate level
    if ! [[ "$target_level" =~ ^[1-6]$ ]]; then
        echo "Error: Level must be between 1 and 6" >&2
        return 1
    fi

    # Generate the regex pattern for the level (e.g., ^### for level 3)
    local pattern="^"
    for i in $(seq 1 "$target_level"); do
        pattern="${pattern}#"
    done
    pattern="${pattern} "

    # Get all line numbers for headers at this level
    grep -n "$pattern" "$file" | while IFS=: read -r line_num content; do
        # Verify exact level match (not a higher level like ####)
        local level=$(echo "$content" | sed -E 's/^(#+).*/\1/' | tr -d '\n' | wc -c)
        if [ "$level" -eq "$target_level" ]; then
            # Extract header text
            local text=$(echo "$content" | sed -E 's/^#+\s*//' | sed -E 's/\s*$//')
            echo "=== $text ==="
            echo ""
            extract_section "$file" "$text"
            echo ""
        fi
    done
}

# Main command dispatch
case "$COMMAND" in
    list)
        list_headers "$FILE"
        ;;
    extract)
        if [ $# -lt 3 ]; then
            echo "Error: extract requires a header name" >&2
            usage
        fi
        extract_section "$FILE" "$3"
        ;;
    extract-level)
        if [ $# -lt 3 ]; then
            echo "Error: extract-level requires a level number (1-6)" >&2
            usage
        fi
        extract_by_level "$FILE" "$3"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'" >&2
        usage
        ;;
esac
