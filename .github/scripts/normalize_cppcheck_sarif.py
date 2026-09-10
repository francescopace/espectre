#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Prepare Cppcheck findings for upload, retaining reviewed exclusions in the raw report."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.report.resolve() == args.output.resolve():
        parser.error('--output must differ from --report to preserve the raw analysis')
    # Never leave a previous upload report behind if normalization fails.
    args.output.unlink(missing_ok=True)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    report = json.loads(args.report.read_text(encoding='utf-8'))
    for run in report['runs']:
        results = []
        notes = []
        seen = set()
        excluded = 0
        duplicates = 0
        for result in run.get('results', []):
            for location in result.get('locations', []) + result.get('relatedLocations', []):
                physical = location.get('physicalLocation', {})
                region = physical.get('region', {})
                # Cppcheck 2.17.1 uses line/column zero for file-level notes.
                # SARIF coordinates start at one; omit the region, keeping the file.
                if region.get('startLine') == 0:
                    physical.pop('region')
                else:
                    for coordinate in ('startColumn', 'endColumn', 'endLine'):
                        if region.get(coordinate) == 0:
                            region.pop(coordinate)
            rule = result['ruleId']
            if rule == 'normalCheckLevelMaxBranches':
                notes.append({
                    'descriptor': {'id': rule},
                    'level': 'note',
                    'message': result['message'],
                    'locations': result.get('locations', []),
                })
                continue
            # Later locations describe the call chain, not the finding's primary file.
            locations = result.get('locations', [])
            file = (locations[0].get('physicalLocation', {}).get('artifactLocation', {}).get('uri')
                    if locations else None)
            if rule in config['style_rules'] or any(
                rule == entry['rule'] and file == entry['file']
                and result['message']['text'] in entry['messages']
                for entry in config['reviewed_findings']
            ):
                excluded += 1
                continue
            key = json.dumps([rule, result['message'], result.get('locations', [])], sort_keys=True)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            results.append(result)
        run['results'] = results
        run.setdefault('properties', {})['qualityReport'] = {
            'excludedFindings': excluded,
            'duplicateFindings': duplicates,
        }
        # Parser failures must not be summarized as a successful analysis.
        complete = not any(result['ruleId'] in {
            'syntaxError', 'unknownMacro', 'internalError', 'preprocessorErrorDirective',
        } for result in results)
        run.setdefault('invocations', []).append({
            'executionSuccessful': complete,
            'toolExecutionNotifications': notes,
        })
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
