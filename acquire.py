"""
A module for obtaining repo readme and language data from the github API.

Before using this module, read through it, and follow the instructions marked TODO.

After doing so, run it like this:

    python acquire.py

To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.



headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )

#### FUNCTION TO GET REPO LINKS
def get_repo_links():
    '''
    NOTE!!! VERY SLOW. IF DON'T HAVE A JSON FILE MAKE SURE TO RUN THIS FUNCTION AT LEAST FOR 1 HR
    
    Scraps the links of the repositories and saves them to the list
    '''
    filename = 'REPOS.json'
    REPOS=[]
    #headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}
    languages = ['JavaScript', 'Python', 'C#', 'Java']
    # if the json file is available
    if os.path.isfile(filename):
        # read from json file
        with open(filename, "r") as json_file:
            REPOS = json.load(json_file)
    else:
        for i in range(1, 100):
            print(i)
            if i == 1:
                start_link = 'https://github.com/search?q=space&type=Repositories'
            else:
                start_link = f'https://github.com/search?p={i}&q=space&type=Repositories'

            response = requests.get(start_link, headers=headers)
            if response.status_code != 200:
                print('problem' + str(response.status_code))
                sleep(30)
                response = requests.get(start_link, headers=headers)
            print(response.status_code)
            soup = BeautifulSoup(response.content, 'html.parser')

            all_blocks = soup.find_all('li', class_='repo-list-item hx_hit-repo d-flex flex-justify-start py-4 public source')
            if type(all_blocks) == None:
                print('all blocks fail')
                sleep(30)
                all_blocks = soup.find_all('li', class_='repo-list-item hx_hit-repo d-flex flex-justify-start py-4 public source')
            for block in all_blocks:
                try:
                    language = block.find('span', itemprop='programmingLanguage').text
                except:
                    continue
                if language in languages:
                    link = block.find('a', class_='v-align-middle')['href'][1:]
                    REPOS.append(link)
            sleep(20)
        
        with open(filename, "w") as outfile:
            json.dump(REPOS, outfile)
    return REPOS

REPOS = get_repo_links()


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        return repo_info["language"]
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    WARNING!!! VERY SLOW. IF DON'T HAVE A JSON FILE MAKE SURE TO RUN THIS FUNCTION AT LEAST FOR 1 HR

    Loop through all of the repos and process them. Returns the processed data.
    """
    if os.path.isfile('data.json'):
        # read from json file
        with open('data.json', "r") as json_file:
            data = json.load(json_file)
    else:
        data = [process_repo(repo) for repo in REPOS]
        with open('data.json', "w") as outfile:
            json.dump(data, outfile)
    return data


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)
