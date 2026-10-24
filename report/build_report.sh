#!/usr/bin/env bash
# Rebuild the report PDF and DOCX from REPORT.md (needs pandoc; PDF via Chromium/Playwright).
set -euo pipefail
cd "$(dirname "$0")"
pandoc REPORT.md -s --mathml --css report.css --resource-path=. -o report.html
node pdf.js "$PWD/report.html" "$PWD/MMA3001_Project_Report_Harvey.pdf"
pandoc REPORT.md -o MMA3001_Project_Report_Harvey.docx --resource-path=.:..
