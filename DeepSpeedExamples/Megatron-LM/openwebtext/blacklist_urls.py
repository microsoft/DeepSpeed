# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import glob
import re
import time
import tldextract
import sys


# List of the domains to blacklist.
domain_blacklist = set([
    '500px',
    'aapks',
    'akamaihd',
    'amazon',
    'apple',
    'artifactfire',
    'artstation',
    'awwni',
    'bandcamp',
    'battleforthenet',
    'coinscalendar',
    'dailymotion',
    'deviantart',
    'discord',
    'discordapp',
    'dlapkandroid',
    'dropbox',
    'e621',
    'ebay',
    'edealinfo',
    'erome',
    'eroshare',
    'explosm',
    'facebook',
    'fbcdn',
    'flickr',
    'furaffinity',
    'futhead',
    'gatopardo',
    'gfycat',
    'gifsound',
    'gifsoup',
    'giphy',
    'github',
    'google',
    'gunprime',
    'gyazo',
    'hotdealstar',
    'imagefap',
    'imageshack',
    'imgflip',
    'imgur',
    'instagram',
    'karmadecay',
    'kryptocal',
    'kym-cdn',
    'liveleak',
    'livememe',
    'lmgtfy',
    'magaimg',
    'memegenerator',
    'minorplanetcenter',
    'minus',
    'mobafire',
    'morejpeg',
    'nocookie',
    'pcpartpicker',
    'photobucket',
    'pinimg',
    'pinterest',
    'pixiv',
    'pornhub',
    'prntscr',
    'puu',
    'qkme',
    'quickmeme',
    'radd',
    'redd',
    'reddit',
    'reddit-stream',
    'redditlog',
    'redditmedia',
    'reddituploads',
    'redtube',
    'reupp',
    'reverb',
    'roanoke',
    'rollingstone',
    'sli',
    'soundcloud',
    'soundgasm',
    'spankbang',
    'spotify',
    'strawpoll',
    'streamable',
    'timeanddate',
    'tinypic',
    'touhouradio',
    'tumblr',
    'twimg',
    'twitch',
    'twitter',
    'vid',
    'vimeo',
    'vine',
    'vkaao',
    'vocaroo',
    'voyagefusion',
    'walmart',
    'wciu',
    'wikimedia',
    'wikipedia',
    'xhamster',
    'xkcd',
    'xvideos',
    'youtu',
    'youtube',
    'youtubedoubler',
    'ytimg',
    'zillexplorer',
])

def domain_is_in_blacklist(url):
    domain = tldextract.extract(url).domain
    return domain in domain_blacklist


# List of extentions to blacklist.
extentions_blacklist = (
    '.3gp',
    '.7z'
    '.ai',
    '.aif',
    '.apk',
    '.app',
    '.avi',
    '.bin',
    '.bmp',
    '.bz2',
    '.css',
    '.csv',
    '.dat',
    '.deb',
    '.dmg',
    '.doc',
    '.docx',
    '.exe',
    '.gif',
    '.gifv',
    '.gz',
    '.iso',
    '.jar',
    '.jpeg',
    '.jpg',
    '.js',
    '.log',
    '.mid',
    '.midi',
    '.mkv',
    '.mov',
    '.mp3',
    '.mp4',
    '.mpeg',
    '.mpg',
    '.ogg',
    '.ogv',
    '.otf',
    '.pdf',
    '.pkg',
    '.png',
    '.pps',
    '.ppt',
    '.pptx',
    '.psd',
    '.py',
    '.qt',
    '.ram',
    '.rar',
    '.sql',
    '.svg',
    '.swf',
    '.tar.gz',
    '.tar',
    '.tgz',
    '.tiff',
    '.ttf',
    '.txt',
    '.wav',
    '.webm',
    '.wma',
    '.wmv',
    '.xls',
    '.xlsx',
    '.xml',
    '.xz',
    '.zip',
)

def extention_is_in_blacklist(url):
    if url.split('?')[0].lower().endswith(extentions_blacklist):
        return True
    return False


# Malformed urls.
# This function is adapted from:
#   https://stackoverflow.com/questions/7160737/python-how-to-validate-a-url-in-python-malformed-or-not
url_regex = re.compile(
    r'^(?:http)s?://' # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
    r'(?::\d+)?' # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)
def url_is_malformed(url):
    return re.match(url_regex, url) is None


def print_progress(prefix, start_time, urls_counter,
                   domain_blacklist_counter,
                   extention_blacklist_counter,
                   short_url_counter, malformed_url_counter,
                   duplicate_url_counter):
    string = prefix + ' | '
    string += 'time elapsed (s): {:.2f} | '.format(time.time() - start_time)
    string += 'number of urls: {} | '.format(urls_counter)
    string += 'domain blacklisted: {} | '.format(domain_blacklist_counter)
    string += 'extention blacklisted: {} | '.format(extention_blacklist_counter)
    string += 'short urls (<=8): {} | '.format(short_url_counter)
    string += 'malformed urls: {} | '.format(malformed_url_counter)
    string += 'duplicate urls: {}'.format(duplicate_url_counter)
    print(string, flush=True)


if __name__ == '__main__':


    print('remove blacklisted urls ..')

    # Path to the url files.
    path = sys.argv[1]
    # Output url file.
    output = sys.argv[2]

    # Get the list of url files.
    files = glob.glob(path + '/*.txt')
    print('> found {} files'.format(len(files)))

    urls = set()
    urls_counter = 0
    domain_blacklist_counter = 0
    extention_blacklist_counter = 0
    short_url_counter = 0
    malformed_url_counter = 0
    duplicate_url_counter = 0
    start_time = time.time()
    for filename in files:
        with open(filename, 'r') as f:
            for line in f:
                url = line.strip()
                urls_counter += 1
                if domain_is_in_blacklist(url):
                    print('[DOMAIN BLACKLIST]: {}'.format(url), flush=True)
                    domain_blacklist_counter += 1
                elif extention_is_in_blacklist(url):
                    print('[EXTENTION BLACKLIST]: {}'.format(url), flush=True)
                    extention_blacklist_counter += 1
                elif len(url) <= 8:
                    print('[SHORT URL]: {}'.format(url), flush=True)
                    short_url_counter += 1
                elif url_is_malformed(url):
                    print('[MALFORMED URL]: {}'.format(url), flush=True)
                    malformed_url_counter += 1
                elif url in urls:
                    print('[DUPLICATE URL]: {}'.format(url), flush=True)
                    duplicate_url_counter += 1
                else:
                    urls.add(url)
                if urls_counter % 100000 == 0:
                    print_progress('PROGRESS', start_time, urls_counter,
                                   domain_blacklist_counter,
                                   extention_blacklist_counter,
                                   short_url_counter, malformed_url_counter,
                                   duplicate_url_counter)

    print_progress('FINAL', start_time, urls_counter,
                   domain_blacklist_counter,
                   extention_blacklist_counter,
                   short_url_counter, malformed_url_counter,
                   duplicate_url_counter)

    # Write the final set of urls.
    print('> writing cleaned up url list to {}'.format(output))
    with open(output, 'w') as f:
        for url in urls:
            f.write(url + '\n')

    print('done :-)')
