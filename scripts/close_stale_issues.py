"""
Script to mark and close stale issues

Taken and adapted from allennlp:
https://github.com/allenai/allennlp/blob/a63e28c24d091fb371011f644fec3ebd29bfbb7e/scripts/close_stale_issues.py
"""

from datetime import datetime as dt
import os

from github import Github


LABELS_TO_EXEMPT = ["contributions welcome", "merge when ready", "under development", "help wanted"]


def main():
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("microsoft/DeepSpeed")
    open_issues = repo.get_issues(state="open")

    open_count = 0
    closing_count = 0
    for issue in open_issues:
        open_count += 1
        if (
            issue.milestone is None
            and not issue.assignees
            and issue.pull_request is None
            and (dt.utcnow() - issue.updated_at).days > 7
            and (dt.utcnow() - issue.created_at).days >= 14
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):
            print("Closing", issue)
            closing_count += 1
            # issue.create_comment(
            #     "This issue is being closed due to lack of activity. "
            #     "If you think it still needs to be addressed, please comment on this thread 👇"
            # )
            # issue.add_to_labels("stale")
            # issue.edit(state="closed")
    print(f'closing_count={closing_count}, open_count={open_count}')

if __name__ == "__main__":
    main()
