"""Export dated Git evidence without rewriting RELIC authorship or inferring eligibility."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess


def audit(root):
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(root), *arguments], text=True)
    entries = []
    event_start = datetime.fromisoformat('2026-10-16T00:00:00-04:00')
    for line in git('log', '--all', '--format=%H%x09%P%x09%aI%x09%cI%x09%an%x09%s', '--reverse').splitlines():
        sha, parents, author_time, committer_time, author, subject = line.split('\t', 5)
        paths = git('diff-tree', '--root', '--no-commit-id', '--name-only', '-r', sha).splitlines()
        entries.append({'commit': sha, 'parents': parents.split(), 'author_date': author_time,
            'committer_date': committer_time, 'author': author, 'subject': subject, 'paths': paths,
            'recorded_period': 'before_event' if datetime.fromisoformat(author_time) < event_start else 'during_or_after_event',
            'merge': len(parents.split()) > 1})
    return {'schema': 'viper.git_provenance.v1', 'historical_project': 'RELIC', 'project': 'VIPER',
            'head': git('rev-parse', 'HEAD').strip(), 'commits': entries,
            'refs': git('for-each-ref', '--format=%(refname) %(objectname)').splitlines(),
            'eligibility_verified': False,
            'limitations': ['Git dates alone do not prove when code was originally written.',
                           'Merges retain parent ancestry; imported and cherry-picked features need origin attribution.',
                           'This audit does not confirm event registration, ownership or licensing.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as stream: json.dump(audit(Path(__file__).resolve().parents[1]), stream, indent=2)
