from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time
from seleniumrequests import Chrome, PhantomJS
import requests
import dateutil.parser
import redis
import json
import logging
import argparse
from tqdm import *
from datetime import datetime
import pytz
import random

root_url = "https://api.fanduel.com"
driver = None


def login(username, password):
    login_url = "https://www.fanduel.com/p/login#login"
    driver.set_window_size(1024, 768)  # optional
    driver.get(login_url)
    time.sleep(10)
    element = driver.find_element_by_id("ccf1")
    password_el = driver.find_element_by_id("password")
    password_el.send_keys(password)
    email_el = driver.find_element_by_id("email")
    email_el.send_keys(username)
    driver.find_element_by_name("login").click()


def get_api_client_id():
    fd_object = driver.execute_script("return FD")
    api_client_id = fd_object['config']['apiClientId']
    return api_client_id


def get_x_auth_token():
    response = driver.request('GET', 'https://www.fanduel.com/games')
    auth_token = response.cookies['X-Auth-Token']
    return auth_token


def get_api_credentials(username, password):
    api_client_id = None
    auth_token = None
    while api_client_id is None or auth_token is None:
        try:
            login(username, password)
            api_client_id = get_api_client_id()
            auth_token = get_x_auth_token()

        except Exception as e:
            print e
        if api_client_id is not None and auth_token is not None:
            break
        else:
            time.sleep(600)
    return api_client_id, auth_token


def get_contests(api_client_id, auth_token):
    headers = {'X-Auth-Token': auth_token, "Authorization": 'Basic {0}'.format(api_client_id)}
    response1 = requests.get('{0}/contests?include_restricted=true'.format(root_url), headers=headers)
    if response1.status_code != 200:
        logging.error("Got a {0}, trying to reauth".format(response1.status_code))
        login()
    return response1.json()


# contest_id looks like 14446-21856479
def get_contest(contest_id, api_client_id, auth_token):
    headers = {'X-Auth-Token': auth_token, "Authorization": 'Basic {0}'.format(api_client_id)}
    response1 = requests.get('{0}/contests/{1}'.format(root_url, contest_id), headers=headers)
    return response1.json()


# contest_timing_id lokes like 14446
def get_players(contest_timing_id, api_client_id, auth_token):
    headers = {'X-Auth-Token': auth_token, "Authorization": 'Basic {0}'.format(api_client_id)}
    response1 = requests.get('{0}/fixture-lists/{1}/players'.format(root_url, contest_timing_id), headers=headers)
    return response1.json()


def get_fixtures(contest_timing_id, api_client_id, auth_token):
    headers = {'X-Auth-Token': auth_token, "Authorization": 'Basic {0}'.format(api_client_id)}
    response1 = requests.get('{0}/fixture-lists/{{1}}'.format(root_url, contest_timing_id), headers=headers)
    return response1.json()


def init_driver():
    global driver
    user_agent = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_4) " + "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.57 Safari/537.36")
    dcap = dict(DesiredCapabilities.PHANTOMJS)
    dcap["phantomjs.page.settings.userAgent"] = user_agent
    driver = PhantomJS(executable_path="/usr/local/bin/phantomjs", desired_capabilities=dcap)


def process_contests():
    api_client_id, auth_token = get_api_credentials("james.ostheimer@gmail.com", 'Rlaaooc1')
    timestamp = int(time.time())
    contests = get_contests(api_client_id, auth_token)
    logging.info("Processing {0} contests:".format(len(contests['contests'])))
    for contest in tqdm(contests['contests']):
        contest_id = contest['id']
        contest_details = get_contest(contest_id, api_client_id, auth_token)
        contest_status_started = contest_details['fixture_lists'][0]['status']['started']
        contest_status_final = contest_details['fixture_lists'][0]['status']['final']
        if not contest_status_final:
            if not r.exists('fanduel::final::{0}'.format(contest_id)):
                r.set('fanduel::final::{0}'.format(contest_id), contest_details['fixture_lists'][0]['start_date'])


def fix_contests():
    api_client_id, auth_token = get_api_credentials("james.ostheimer@gmail.com", 'Rlaaooc1')
    for key in tqdm(r.keys("fanduel::contest::*")):
        contest_id = key.split("::")[2]
        contest_timing_id = contest_id.split("-")[0]
        if 'error' in json.loads(r.get("fanduel::contest::{0}".format(key))):
            players = get_players(contest_timing_id, api_client_id, auth_token)
            fixtures = get_fixtures(contest_timing_id, api_client_id, auth_token)
            r.set('fanduel::contest::players::{0}'.format(contest_id), json.dumps(players))
            r.set('fanduel::contest::fixtures::{0}'.format(contest_id), json.dumps(fixtures))


def check_on_contests():
    api_client_id, auth_token = get_api_credentials("james.ostheimer@gmail.com", 'Rlaaooc1')
    logging.info("Checking status of {0} contests".format(len(r.keys("fanduel::final::*"))))
    number_final = 0
    for key in tqdm(r.keys("fanduel::final::*")):
        contest_id = key.split("::")[2]
        start_date = dateutil.parser.parse(r.get(key))
        # be a good steward and don't call unless we are 8 hours after the start date
        time_diff = datetime.utcnow().replace(tzinfo=pytz.utc) - start_date
        if time_diff.total_seconds() > 8 * 60 * 60:
            contest_details = get_contest(contest_id, api_client_id, auth_token)
            if 'fixture_lists' not in contest_details:
                continue
            contest_status_final = contest_details['fixture_lists'][0]['status']['final']
            # contest is final, store the contest data, players and fixtures
            if contest_status_final:
                number_final += 1
                contest_timing_id = contest_id.split("-")[0]
                players = get_players(contest_timing_id, api_client_id, auth_token)
                fixtures = get_fixtures(contest_timing_id, api_client_id, auth_token)
                r.set('fanduel::contest::{0}'.format(contest_id), json.dumps(contest_details))
                r.set('fanduel::contest::players::{0}'.format(contest_id), json.dumps(players))
                r.set('fanduel::contest::fixtures::{0}'.format(contest_id), json.dumps(fixtures))
                r.delete(key)
    logging.info("Processed {0} final contests".format(number_final))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', dest='log_file', default=None, help='set log file')
    parser.add_argument('--logging_level', dest='logging_level', default=logging.INFO,
                        help='logging level: CRITICAL=50 ERROR=40 WARNING=30 INFO=20 DEBUG=10 NONE=0')
    args = parser.parse_args()

    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file, level=args.logging_level,
                            format='%(asctime)s %(name)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=args.logging_level, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    logging.root.setLevel(args.logging_level)
    logging.getLogger("requests").setLevel(logging.WARNING)
    init_driver()
    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    # runs through all the contests and gets players and fixtures without regard to how old or anything
    # fix_contests()
    while 1:
        logging.info("Running a run")
        process_contests()
        check_on_contests()
        time.sleep(random.normalvariate(45. * 60, 5 * 60))
