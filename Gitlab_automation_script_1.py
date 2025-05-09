import os
import shutil
from git import Repo, GitCommandError
import requests
from datetime import datetime
import sys

def get_files_changed_in_commit(repo, commit_hash):
    """
    Get the list of files added or modified in the specified commit.
    Excludes deleted files.
    """
    try:
        # Fetch only added (A) and modified (M) files
        changed_files = repo.git.diff_tree("--no-commit-id", "--name-status", "-r", commit_hash).splitlines()
        #if you need to fetch the deleted files also, omit this step
        filtered_files = [
            line.split('\t')[1] for line in changed_files if line.startswith(("A", "M"))
        ]
        return filtered_files
    except GitCommandError as gce:
        print(f"Git command failed: {gce}")
        return []

def get_latest_commit(repo):
    """
    Fetches the latest commit from the repository and returns its details.
    """
    try:
        commits = list(repo.iter_commits('main'))
        latest_commit = commits[0]
        return latest_commit.hexsha
    except Exception as e:
        print(f"Error while fetching commit details: {e}")
        return None

def sync_recent_files(user_id, token, cert_path=None, branch="main", repo_type="ALL"):
    try:
        # Configuration based on the repo type
        repo_config = {
            "FCM": {
                "repo1_url": f"https://{user_id}:{token}@gitlab.dell.com/ebia/ebia-cicd/ebia-marketing/mktg-data-ba/digital/archive/fcm.git",
                "repo2_url": f"https://{user_id}:{token}@gitlab.dell.com/ebia/ebia-cicd/ebia-marketing/mktg-data-ba/digital/archive/target_repo_fcm.git",
                "base_path": "src/main/resources/common/ingestion-base-share/PINE/ScriptFiles/FCM",
                "folders_to_sync": ["app", "config", "shell"],
                "project_name": "ebia/ebia-cicd/ebia-marketing/mktg-data-ba/digital/archive/target_repo_fcm",
            },
            "EIG": {
                "repo1_url": f"https://{user_id}:{token}@gitlab.dell.com/ebia/ebia-cicd/ebia-marketing/mktg-data-ba/digital-tg/cep_eig.git",
                "repo2_url": f"https://{user_id}:{token}@gitlab.dell.com/ebia/ebia-cicd/ebia-marketing/mktg-data-ba/digital-tg/archive/cep_eig.git",
                "base_path": None,
                "folders_to_sync": ["PBE_files", "config_files", "graph_jobs"],
                "project_name": "ebia/ebia-cicd/ebia-marketing/mktg-data-ba/digital-tg/archive/cep_eig",
            },
            "DKR": {
                "repo1_url": f"https://{user_id}:{token}@gitlab.dell.com/ebia/ebia-cicd/ebia-marketing/mktg-data-ba/digital/archive/pine.git",
                "repo2_url": f"https://{user_id}:{token}@gitlab.dell.com/ebia/ebia-cicd/ebia-marketing/mktg-data-ba/digital/archive/target_repo_dkr.git",
                "base_path": "src/main/resources/common/ingestion-base-share/PINE",
                "folders_to_sync": ["GPListFiles", "SqlFiles"],
                "project_name": "ebia/ebia-cicd/ebia-marketing/mktg-data-ba/digital/archive/target_repo_dkr",
            },
        }

        if repo_type == "ALL":
            for repo in repo_config.keys():
                print(f"\nRunning synchronization for repository type: {repo}")
                sync_recent_files(user_id, token, cert_path, branch, repo)
            return

        if repo_type not in repo_config:
            print(f"Invalid repository type: {repo_type}")
            return

        config = repo_config[repo_type]

        # Temp directories
        repo1_temp_path = "/tmp/repo1"
        repo2_temp_path = "/tmp/repo3"

        # Remove temp directories if they exist
        if os.path.exists(repo1_temp_path):
            shutil.rmtree(repo1_temp_path)
        if os.path.exists(repo2_temp_path):
            shutil.rmtree(repo2_temp_path)

        # Clone repositories
        repo1 = Repo.clone_from(config["repo1_url"], repo1_temp_path, branch=branch)
        repo2 = Repo.clone_from(config["repo2_url"], repo2_temp_path, branch=branch)

        # Get latest commit from Repo 1
        latest_commit_hash = get_latest_commit(repo1)
        if not latest_commit_hash:
            print("Failed to fetch the latest commit.")
            return

        print(f"Latest commit hash: {latest_commit_hash}")

        # Get files changed in the latest commit
        changed_files = get_files_changed_in_commit(repo1, latest_commit_hash)
        if not changed_files:
            print("No files changed in the latest commit.")
            return

        # Filter changed files to include only those in the folders to sync
        filtered_files = [
            f for f in changed_files
            if any(f.startswith(folder + "/") for folder in config["folders_to_sync"])
        ]

        if not filtered_files:
            print("No changed files in the required folders to sync.")
            return

        # Read exclusion list from cicd_exclude.txt in Repo 2
        exclude_file_path = os.path.join(repo2_temp_path, "cicd_exclude.txt")
        excluded_filenames = set()
        if os.path.exists(exclude_file_path):
            with open(exclude_file_path, "r") as f:
                for line in f:
                    excluded_filenames.update([name.strip() for name in line.split(",") if name.strip()])

        print(f"Excluded filenames: {excluded_filenames}")

        # Create a new branch in Repo 2
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_branch = f"sync_recent_files_{timestamp}"
        repo2.git.checkout("-b", new_branch)

        # Copy filtered files to Repo 2
        for relative_path in filtered_files:
            file_name = os.path.basename(relative_path)  # Extract the file name
            if file_name in excluded_filenames:  # Check if the file name is in the exclusion list
                print(f"Excluding {relative_path} as its filename is listed in cicd_exclude.txt")
                continue

            source_path = os.path.join(repo1_temp_path, relative_path)
            target_path = os.path.join(repo2_temp_path, config["base_path"], relative_path) if config["base_path"] else os.path.join(repo2_temp_path, relative_path)

            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            if os.path.exists(source_path) and os.path.isfile(source_path):
                shutil.copy2(source_path, target_path)
                print(f"Copied {source_path} to {target_path}")

        # Set Git author and committer details for Repo 2
        repo2.config_writer().set_value("user", "name", user_id).release()
        repo2.config_writer().set_value("user", "email", "gokul.subramanian1@dellteam.com").release()

        # Commit changes
        repo2.git.add(all=True)
        repo2.index.commit("Sync recent files for specified folders")

        # Push changes to the new branch
        repo2.git.push("--set-upstream", config["repo2_url"], new_branch)

        # Create a merge request using GitLab API
        project_name = config["project_name"]
        merge_request_url = f"https://gitlab.dell.com/api/v4/projects/{project_name.replace('/', '%2F')}/merge_requests"

        headers = {"PRIVATE-TOKEN": token}
        data = {
            "source_branch": new_branch,
            "target_branch": branch,
            "title": f"Sync recent files - {timestamp}",
        }

        response = requests.post(
            merge_request_url,
            headers=headers,
            json=data,
            verify=cert_path,
        )

        if response.status_code == 201:
            print("Merge request created successfully!")
        else:
            print(f"Failed to create merge request: {response.status_code}, {response.json()}")

        # Clean up temp directories
        shutil.rmtree(repo1_temp_path)
        shutil.rmtree(repo2_temp_path)

        print("Synchronization completed and merge request created.")

    except GitCommandError as gce:
        print(f"Git command failed: {gce}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage 
if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            raise ValueError("Usage: python3 script.py <repo_type>.\nEnter a valid repository name (FCM, EIG, DKR, ALL) for the script to execute")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    repo_type = sys.argv[1]
    user_id = "g_subramanian"
    token = "glpat-M4f6qRpVTDtAqe7aCcKg"
    cert_path = "/home/g_subramanian/updated_gitlab_dell_certificate.crt"

    sync_recent_files(user_id, token, cert_path, repo_type=repo_type)
