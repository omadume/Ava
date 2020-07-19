# All imports - some no longer required

import os  # to store environment variables
import sys  # to read username from terminal
import json  # for json data
from json.decoder import JSONDecodeError  # for print json in readable format
import spotipy  # for Spotify API
import spotipy.util as util  # for Spotify API
from spotipy import Spotify, oauth2  # for Spotify Authentication process
from spotipy.oauth2 import SpotifyClientCredentials  # for using Spotify Web Api for developers - only need if using client credentials flow
from spotipy.client import SpotifyException  # for catching / handling exceptions
import webbrowser  # to open urls in browser if needed
from textblob import TextBlob  # for NLP tools / methods
from textblob.np_extractors import ConllExtractor  # for noun phrase extraction
from textblob.classifiers import NaiveBayesClassifier  # for intent classification
import csv  # for using csv files
import re  # for regular expressions
import string  # for text pre-processing "cleaning"
import unidecode  # for text pre-processing "cleaning" - removing diacritical marks
from random import randint  # for getting random integers, for randomisation of search results
import speech_recognition as sr  # for google speech-to-text
from gtts import gTTS  # for google text-to-speech
from pygame import mixer  # for audio player
import time  # for time delays
from textblob import Word  # for getting synsets of words in semantic similarity method
from textblob.wordnet import VERB,ADV,ADJ,NOUN  # WordNet only has synsets for these types

# Spotify Authentication Setup

username = 'XXXX'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'XXXX'
client_id = 'XXXX'
client_secret = 'XXXX'
redirect_uri = 'http://google.co.uk/'
scope = 'user-modify-playback-state, user-read-playback-state'  # the scopes for which access to user data is required

token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
if token:
    print("token: ", token)
    spotifyObject = spotipy.Spotify(auth=token)

else:
    print("Can't get token for", username)

count = 0  # used in 'speak' method for switching between 2 files for creation of audio output - prevents read/write error


def speak(audio_string):
    """
    This method converts Python text strings to audio output which the bot "speaks" back to the user.

    :param audio_string: the text string to convert to audio
    :type audio_string: string
    :return none
    """
    global count
    print(audio_string)
    tts = gTTS(text=audio_string, lang='en')
    tts.save(f'audio{count%2}.mp3')
    mixer.init()
    while mixer.music.get_busy():
        time.sleep(1)
    mixer.music.load(f'audio{count%2}.mp3')
    mixer.music.play()
    count += 1
    global spoken
    spoken = True  # to check if a response has been returned

    return

# Introductory message and bot usage tips for user

speak("\nHello, I'm Ava. I'm your friendly assistant for all things Spotify music related")
print("You can ask me for things like song suggestions for genres you'd like to listen to, or moods you're in, or perhaps an activity you're doing.")
print("\n--Helpful Tips--\n")
print("- When referring to multi-word genres like R&B, Synthpop, Heavy Metal, and Rock and roll, try switching to text mode, and typing it with hyphens:")
print("e.g. r-n-b, synth-pop, heavy-metal, rock-n-roll")
print("\n- When referring to the specific name of something with multiple words (e.g. To pimp a butterfly by Kendrick), try putting it single quotes:")
print("e.g. 'To pimp a butterfly' by Kendrick (while in text mode) - this helps me to focus on it more accurately.")
print("\n----------------")
speak("\n...So how can I help you today?")
print("(Remember: Whenever you have no more questions, just say or type 'Done'.)")


def conversion_to_wordnet_tag(tag):
    """
    This method converts TextBlob POS tag format to WordNet style.

    :param tag: the TextBlob tag to convert to WordNet tag
    :type tag: string
    :return wordnet_tag
    :rtype string
    """
    wordnet_tag = None
    if tag.startswith('N'):  # noun
        wordnet_tag = 'n'
    if tag.startswith('V'):  # verb
        wordnet_tag = 'v'
    if tag.startswith('J'):  # adjective
        wordnet_tag = 'a'
    if tag.startswith('R'):  # adverb
        wordnet_tag = 'r'

    return wordnet_tag


def obtain_synset(word, tag):
    """
    This method obtains the synset of a word.

    :param word: the TextBlob word to obtain the synset of
    :type word: string
    :param tag: the TextBlob tag to use in obtaining the word synset
    :type tag: string
    :return synset
    :rtype list
    """
    synset = None
    if tag is not None:
        try:
            synset = Word(word).get_synsets(pos=tag)[0]  # synsets are ordered by how frequent that sense appears in the WordNet corpus - most frequent is first
        except:
            pass  # if no synsets found for word, do nothing
    return synset


def classification_based_on_semantic_similarity(input_text):
    """
    This method classifies user input text using semantic similarity.

    :param input_text: the user input which is to be classified
    :type input_text: string
    :return semantic_classification_label
    :rtype string
    """
    with open('IntentClassifier.csv', encoding='latin-1') as f:
        classifier_data = [tuple(row) for row in csv.reader(f)]
    sentences = []
    for item, item_label in classifier_data:
        sentence = TextBlob(item)
        sentences.append(sentence)

    best_score = 0
    most_similar = ""
    for sentence in sentences:
        score = semantic_similarity(input_text, sentence)
        if score > best_score:
            best_score = score
            most_similar = sentence

    semantic_classification_label = ""
    for item, item_label in classifier_data:
        if item == most_similar:
            semantic_classification_label = item_label

    return semantic_classification_label


def semantic_similarity(sentence1, sentence2):
    """
    This method produces the semantic similarity score of two sentences.

    :param sentence1: the first sentence to use in comparing semantic similarity
    :type sentence1: string
    :param sentence2: the second sentence to use in comparing semantic similarity
    :type sentence2: string
    :return score
    :rtype int
    """
    sentence1 = TextBlob(sentence1)
    s1_synsets = []
    s2_synsets = []
    for word, tag in sentence1.tags:
        s1_synsets.append(obtain_synset(word, conversion_to_wordnet_tag(tag)))

    for word, tag in sentence2.tags:
        s2_synsets.append(obtain_synset(word, conversion_to_wordnet_tag(tag)))

    s1_synsets_cleaned = []
    s2_synsets_cleaned = []

    # Filtering out type Nones
    for s in s1_synsets:
        if s is not None:
            s1_synsets_cleaned.append(s)
    for s in s2_synsets:
        if s is not None:
            s2_synsets_cleaned.append(s)

    score = 0.0
    score_count = 0
    for synset in s1_synsets_cleaned:
        scores = []
        for s in s2_synsets_cleaned:
            similarity_score = synset.path_similarity(s)
            if similarity_score is not None:  # filtering out potential NoneType - if no path between the 2 synsets
                scores.append(similarity_score)
        if len(scores) > 0:
            best_score = max(scores)  # gets the similarity value of the most similar word in the other sentence (i.e. sentence1)
            score += best_score
            score_count += 1  # keeps track of how many scores have been added
    score /= score_count  # gets average of values

    return score


def symmetric_sentence_similarity(sentence1, sentence2):
    """
    This method turns semantic similarity into a symmetrical function.

    :param sentence1: the first sentence to use in comparing semantic similarity
    :type sentence1: string
    :param sentence2: the second sentence to use in comparing semantic similarity
    :type sentence2: string
    :return score
    :rtype int
    """
    return (semantic_similarity(sentence1, sentence2) + semantic_similarity(sentence2, sentence1)) / 2


def entity_linking_for_named_entities(named_entities_list):
    """
    This method performs entity linking for items identified as 'named entities'.

    :param named_entities_list: the list containing the named entities to link
    :type named_entities_list: list
    :return classifications
    :rtype dict
    """
    with open('SpotifyFeaturesClassifier.csv', encoding='latin-1') as f:
        classifier_data = [tuple(row) for row in csv.reader(f)]

    classifications = {}

    for named_entity in named_entities_list:
        track_names = []
        artist_names = []
        album_names = []
        search_name = named_entity

        for data_name, data_name_type in classifier_data:
            if search_name.lower() == data_name.lower():  # if exact match
                if data_name_type == "track":
                    track_names.append(data_name)
                elif data_name_type == "album":
                    album_names.append(data_name)
                else:
                    artist_names.append(data_name)

        if len(artist_names) == 0 and len(track_names) == 0 and len(album_names) == 0:  # otherwise look for possibilities

            for data_name, data_name_type in classifier_data:

                strings_in_brackets = []

                search_name_set = set(search_name.lower().split())  # creating a set which is hashable - therefore able to be used with '.issubset' method
                data_name_set = set(data_name.lower().split())  # case sensitive so making both lower, using .split() so it matches e.g. only 'will' in will or will smith, but not in william
                if search_name_set.issubset(data_name_set):
                    if data_name_type == "track" or data_name_type == "album":
                        for brackets_string in re.findall(r"\(.*?\)", data_name):  # checks for name as (feat.), etc as this should go towards count of name as an artist not a track
                            strings_in_brackets.append(brackets_string)
                        for brackets_string in re.findall(r"\[.*?\]", data_name):
                            strings_in_brackets.append(brackets_string)

                        brackets_removed = data_name.lower()
                        for strings in strings_in_brackets:
                            strings = strings.lower()  # need to make lowercase as .replace() method is case-sensitive
                            brackets_removed = brackets_removed.replace(strings, "")
                        brackets_removed_set = set(brackets_removed.split())

                        found = False  # checks if search_name found in brackets
                        if len(strings_in_brackets) != 0:
                            for brackets_string in strings_in_brackets:
                                brackets_string_set = set(brackets_string.lower().split())
                                if search_name_set.issubset(brackets_string_set):
                                    artist_names.append(data_name)
                                    found = True
                            if found and search_name_set.issubset(brackets_removed_set):  # if data_name had brackets and search_name is both in and outside of them - might be both a track name and artist name
                                track_names.append(data_name)
                            elif not found:  # if data_name has brackets but search_name was not found in them
                                if data_name_type == "track":
                                    track_names.append(data_name)
                                else:
                                    album_names.append(data_name)

                        elif len(strings_in_brackets) == 0:
                            if data_name_type == "track":
                                track_names.append(data_name)
                            else:
                                album_names.append(data_name)
                    else:
                        artist_names.append(data_name)

        lists = [artist_names, album_names, track_names]
        iterable = iter(lists)
        length = len(next(iterable))
        if all(len(l) == length for l in iterable):  # if all list lengths are equal, favour label 'artist'
            classifications[search_name] = "artist"

        else:
            highest_count = max(len(artist_names), len(album_names), len(track_names))
            index_of_highest_count = 0
            for l in lists:
                if len(l) == highest_count:
                    index_of_highest_count = lists.index(l)

            if index_of_highest_count == 0:
                classifications[search_name] = "artist"
            elif index_of_highest_count == 1:
                classifications[search_name] = "album"
            else:
                classifications[search_name] = "track"

    return classifications


def entity_linking_for_other_meaningful_info(other_meaningful_info):
    """
    This method performs entity linking for items identified as 'other meaningful info'.

    :param other_meaningful_info: the list containing the other meaningful info items to link
    :type other_meaningful_info: list
    :return classifications
    :rtype dict
    """
    with open('OtherFeaturesClassifier.csv', encoding='latin-1') as f:
        classifier_data = [tuple(row) for row in csv.reader(f)]
    classifications = {}
    genre_names = []
    activity_interest_names = []
    mood_names = []
    other_info = []  # e.g. descriptors such as "most popular"

    for info in other_meaningful_info:
        search_name = info

        for data_name, data_name_type in classifier_data:
            if search_name.lower() == data_name.lower():
                if data_name_type == "genre":
                    genre_names.append(data_name)
                    classifications[search_name] = "genre"
                    break
                elif data_name_type == "activity_interest":
                    activity_interest_names.append(data_name)
                    classifications[search_name] = "activity_interest"
                    break
                elif data_name_type == "mood":
                    mood_names.append(data_name)
                    classifications[search_name] = "mood"
                    break
            else:
                other_info.append(search_name)
                classifications[search_name] = "other_info"

    return classifications


def classify_intent(user_input):
    """
    This method classifies the intent of the user input text.

    :param user_input: the user input text to be classified
    :type user_input: string
    :return classification
    :rtype string
    """
    with open('IntentClassifier.csv', 'r') as fp:
        classifier = NaiveBayesClassifier(fp, format="csv")  # classifying intent using Binarized Multinomial Naive Bayes
        classification = classifier.classify(user_input)

    return classification


def extract_named_entities(user_input):
    """
    This method extracts 'named entities' from user input text.

    :param user_input: the user input text to extract named entities from
    :type user_input: string
    :return artists_tracks_albums
    :rtype list
    """
    # targets and extracts named entities by filtering extracted noun phrases based on the POS tags of the words they contain
    artists_tracks_albums = []

    search_query_conllextractor = TextBlob(user_input, np_extractor=ConllExtractor())  # using Conll noun phrase extractor
    for noun_phrase in search_query_conllextractor.noun_phrases:
        np = TextBlob(noun_phrase)
        for np_word, np_word_tag in np.tags:
            for search_query_word, search_query_word_tag in search_query_conllextractor.tags:
                if search_query_word.lower() == np_word.lower():
                    np_word_tag = search_query_word_tag  # gets original POS tag back as breaking down np for use as TextBlob object re-tags np words incorrectly
                    if np_word_tag != "NNP" and np_word_tag != "NNPS" and np_word != "&" and np_word != "+":  # covers things like R&B where '&' would be an np_word:
                        if search_query_word.lower() in np.words:  # checks if word has already been removed e.g. in query "I like music. Give me rap music." - it prevents an error from trying to remove 'music' twice if its only in one extracted noun phrase
                            np.words.remove(search_query_word.lower())  # np.words are lowercase

        if len(np.words) != 0:
            artists_tracks_albums.append(' '.join(np_word for np_word in np.words))

    search_query_fastnpextractor = TextBlob(user_input)  # using FastNP noun phrase extractor - TextBlob's default
    for noun_phrase in search_query_fastnpextractor.noun_phrases:
        np = TextBlob(noun_phrase)
        for np_word, np_word_tag in np.tags:
            for search_query_word, search_query_word_tag in search_query_fastnpextractor.tags:
                if search_query_word.lower() == np_word.lower():
                    np_word_tag = search_query_word_tag
                    if np_word_tag != "NNP" and np_word_tag != "NNPS" and np_word != "&" and np_word != "+":
                        if search_query_word.lower() in np.words:
                            np.words.remove(search_query_word.lower())

        if np.words not in artists_tracks_albums and len(np.words) != 0:
            artists_tracks_albums.append(' '.join(np_word for np_word in np.words))

    for item in artists_tracks_albums:
        if re.match(r"\b([a-z]) (?=[a-z]\b)", item):  # matches words like 'r b' which should be 'r&b' - the '&' was getting removed for some reason in the join, although recognised as an np_word and therefore should have been joined
            item_index = artists_tracks_albums.index(item)
            artists_tracks_albums[item_index] = item.replace(" ", "&")

    artists_tracks_albums = list(dict.fromkeys(artists_tracks_albums))  # deletes any duplicates

    artists_tracks_albums = [item.lower() for item in artists_tracks_albums]  # makes all lowercase

    return artists_tracks_albums


def extract_other_meaningful_info(user_input):
    """
    This method extracts 'other meaningful info' from user input text.

    :param user_input: the user input text to extract other meaningful info items from
    :type user_input: string
    :return genres_activities_interests_moods_descriptors
    :rtype list
    """
    # not using an np extractors, just targeting specific POS tags to try and retrieve meaningful info related to genre, activity_interest and mood
    genres_activities_interests_moods_descriptors = []
    search_query = TextBlob(user_input)  # using TextBlob's default Pattern POS tagger
    count = 0

    for search_query_word, search_query_word_tag in search_query.tags:
        count += 1
        if search_query_word_tag in ["JJ", "VBP", "NN", "CD", "RBS", "JJS", "VBG", "NNS", "VB", "RB"]:  # 'song' sometimes recognised as RB for some reason
            if search_query_word_tag == "JJ" and "RBS" in [word_tag_data for word_tag_data in
                                                           search_query.tags[count - 2]]:  # pairs words like "most recent"
                previous_word = [word_tag_data for word_tag_data in
                                 search_query.tags[count - 2]]  # -2 because index starts at 0
                previous_word = previous_word[:-1]
                genres_activities_interests_moods_descriptors.append(' '.join(previous_word) + " " + search_query_word)
            elif search_query_word_tag == "RBS" and "JJ" in [x for x in search_query.tags[count]]:  # catching e.g. "most recent" again but in reverse, and ignoring the word so as to not duplicate
                continue
            else:
                genres_activities_interests_moods_descriptors.append(search_query_word)

    genres_activities_interests_moods_descriptors = list(dict.fromkeys(genres_activities_interests_moods_descriptors))  # removes any duplicates

    genres_activities_interests_moods_descriptors = [item.lower() for item in genres_activities_interests_moods_descriptors]  # makes all lowercase

    return genres_activities_interests_moods_descriptors


def clean_text(user_input):
    """
    This method performs text pre-processing "cleaning" on user input text.

    :param user_input: the user input text to "clean"
    :type user_input: string
    :return cleaned
    :rtype string
    """
    cleaned = unidecode.unidecode(user_input)  # removes any diactritical marks e.g. accents on letters
    punctuation_to_keep = re.compile(r"(\b[-'&.]\b)|[^\w+$'.]")  # allows in-word -, ', ., and &, and also ' outside words
    cleaned = punctuation_to_keep.sub(lambda m: (m.group(1) if m.group(1) else " "), cleaned)  # removes any punctuation not in 'punctuation_to_keep'
    cleaned = cleaned.strip()  # removes any whitespaces before and after string

    return cleaned


def type_of_music(named_entities_list, other_meaningful_info):
    """
    This method performs search queries for the intent 'type of music'.

    :param named_entities_list: the list of named entities to use for the search query
    :type named_entities_list: list
    :param other_meaningful_info: the list of other meaningful info items to use for the search query
    :type other_meaningful_info: list
    :return none
    """
    albums = []
    songs = []
    playlists = []
    song_uris.clear()  # resetting list for new songs based off new input
    new_items_count = 0  # keeping track of new items in lists
    if len(other_meaningful_info) != 0:

        entity_links = entity_linking_for_other_meaningful_info(other_meaningful_info)

        for entity in entity_links:
            if entity_links.get(entity) == "genre":
                genre = []  # 'genre' has to be passed as a list value to some of the Spotify methods e.g. 'recommendations' method
                genre.append(entity)
                new_items_count += 1

                '''The way albums of a particular genre are found is by using 'search' method to target jazz artists for example, 
                and then getting albums that the artist is on as they are likely to be jazz albums since the artist's predominant genre tag is as such'''
                if "album" in other_meaningful_info or "albums" in other_meaningful_info:

                    try:
                        result = spotifyObject.search(q='genre:' + genre[0], limit=5, offset=randint(0, 15), type='artist', market=None)
                        '''Using an offset of a random number in range 0-15 as 10,000 is max but was bringing very nichely-related genre results 
                        e.g. a japanese hiphop/rap/rnb group for 'pop'. Lowering range of 'randint' offset number brings more closely-related genre results'''

                        if len(result['artists']['items']) != 0:  # preventing KeyError if no artists and therefore no artist 'items'

                            for artist in result['artists']['items']:
                                artist_album = spotifyObject.artist_albums(artist_id=artist['uri'], album_type='album', country=None, limit=1, offset=0)
                                for album in artist_album['items']:
                                    album_name = album['name']
                                    for artist_name in album['artists']:  # get names of all artists on that album
                                        if any(album_name in x for x in albums):
                                            index_of_sublist_with_album = next((x for x, val in enumerate(albums) if album_name in val), None)
                                            albums[index_of_sublist_with_album].insert(len(albums[index_of_sublist_with_album]) - 1, artist_name['name'])
                                        else:
                                            albums.append([artist_name['name'], album_name])

                            if "album" in other_meaningful_info:
                                speak("\nHere's a " + genre[0] + " album by an artist you might like:\n")
                                for album in albums[len(albums)-1:]:
                                    print("{} - {}".format(", ".join(album[:-1]), album[-1]))
                            else:
                                if new_items_count > 1:
                                    speak("\n--The results for " + genre[0] + " have been added to the bottom of the previous list.--")

                                speak("\nHere are some " + genre[0] + " albums by artists you might want to check out:\n")
                                for album in albums:
                                    print("{} - {}".format(", ".join(album[:-1]), album[-1]))

                            genre.pop(0)

                        else:
                            speak("\nSorry, I could not find any " + genre[0] + " albums in the Spotify music library for you.")
                            speak("Spotify only permits me to run searches for a particular list of genres and it may be that this genre is not in the list.")
                            speak("\nTo see the kind of genres I can search for, at any time in our chat, just type 'show genres'.")
                            print("- Tip: For genres like R&B, Synthpop, Heavy Metal, and Rock and roll, try switching to text mode, and typing it like this:")
                            print("r-n-b, synth-pop, heavy-metal, rock-n-roll")
                            print("\n...And if I still can't find anything for you, maybe there just aren't any of your requested search item (ie. 'songs') for that genre currently in the Spotify music library. Sorry.")

                    except SpotifyException:
                            speak("\nSorry, I was unable to make a search with that genre information.")

                elif "playlist" in other_meaningful_info or "playlists" in other_meaningful_info:

                    try:
                        result = spotifyObject.category_playlists(category_id=genre[0], country=None, limit=5, offset=randint(0, 15))

                        if len(result['playlists']['items']) != 0:

                            for playlist in result['playlists']['items']:
                                playlists.append(playlist['name'])

                            if "playlist" in other_meaningful_info:
                                speak("\nHere's a " + genre[0] + " playlist you might like:\n")
                                for playlist in playlists[len(playlists)-1:]:
                                    print(playlist)
                            else:
                                if new_items_count > 1:
                                    speak("\n--The results for " + genre[0] + " have been added to the bottom of the previous list.--")

                                speak("\nHere are some " + genre[0] + " playlists you might want to check out:\n")
                                for playlist in playlists:
                                    print(playlist)

                            genre.pop(0)

                        else:
                            speak("\nSorry, I could not find any " + genre[0] + " playlists in the Spotify music library for you.")
                            speak("Spotify only permits me to run searches for a particular list of genres and it may be that this genre is not in the list.")
                            speak("\nTo see the kind of genres I can search for, at any time in our chat, just type 'show genres'.")
                            print("- Tip: For genres like R&B, Synthpop, Heavy Metal, and Rock and roll, try switching to text mode, and typing it like this:")
                            print("r-n-b, synth-pop, heavy-metal, rock-n-roll")
                            print("\n...And if I still can't find anything for you, maybe there just aren't any of your requested search item (ie. 'songs') for that genre currently in the Spotify music library. Sorry.")

                    except SpotifyException:
                            speak("\nSorry, I was unable to make a search with that genre information.")

                else:  # handles requests for song, songs, and other general terms e.g. find me jazz 'music' or find me pop 'stuff'

                    try:
                        # recommendations method requires seeds as list format
                        result = spotifyObject.recommendations(seed_artists=None, seed_genres=genre,
                                                               seed_tracks=None, limit=5, country=None,
                                                               min_popularity=50, max_popularity=100)  # using min-max instead of target_popularity as it randomizes selection each time which is better
                        if len(result['tracks']) != 0:

                            for track in result['tracks']:
                                for artist_info in track['artists']:
                                    if any(track['name'] in x for x in songs):
                                        index_of_sublist_with_track = next(
                                            (x for x, val in enumerate(songs) if track['name'] in val), None)
                                        songs[index_of_sublist_with_track].insert(
                                            len(songs[index_of_sublist_with_track]) - 1, artist_info['name'])  # catching if song already seen and it is just another artist on the track
                                    else:
                                        songs.append([artist_info['name'], track['name']])

                                song_uris.append(track['uri'])

                            if "song" in other_meaningful_info:
                                speak("\nHere's a popular " + genre[0] + " song by an artist you might like:\n")
                                for song in songs[len(songs)-1:]:
                                    print("1. " + "{} - {}".format(", ".join(song[:-1]), song[-1]))
                                speak("\nIf you'd like to listen to it, just type or say '1'.")
                                print("Note: You can only use this feature if you have Spotify Premium.")
                                print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                            else:  # if no specific singular request made e.g. 'song' then return multiple items
                                if new_items_count > 1:
                                    speak("\n--The results for " + genre[0] + " have been added to the bottom of the previous list.--")

                                speak("\nHere are some popular " + genre[0] + " songs by artists you might want to check out:\n")
                                for song in songs:
                                    song_name = "{} - {}".format(", ".join(song[:-1]), song[-1])
                                    print(str(songs.index(song) + 1) + ". " + song_name)
                                speak("\nIf you'd like to listen to any of them, just type or say their number (e.g. '1').")
                                print("Note: You can only use this feature if you have premium Spotify account.")
                                print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                        else:
                            speak("\nSorry, I could not find any " + genre[0] + " songs in the Spotify music library for you.")
                            speak("Spotify only permits me to run searches for a particular list of genres and it may be that this genre is not in the list.")
                            speak("\nTo see the kind of genres I can search for, at any time in our chat, just type 'show genres'.")
                            print("- Tip: For genres like R&B, Synthpop, Heavy Metal, and Rock and roll, try switching to text mode, and typing it like this:")
                            print("r-n-b, synth-pop, heavy-metal, rock-n-roll")
                            print("\n...And if I still can't find anything for you, maybe there just aren't any of your requested search item (ie. 'songs') for that genre currently in the Spotify music library. Sorry.")

                        genre.pop(0)

                    except SpotifyException:  # if genre not found
                        speak("\nSorry, I was unable to make a search with that genre information.")

            elif entity_links.get(entity) == "mood" or entity_links.get(entity) == "activity_interest":
                mood_activity_interest = entity

                new_items_count += 1
                try:
                    result = spotifyObject.search(q=mood_activity_interest, limit=5, offset=randint(0, 10), type='playlist', market=None)  # didn't use q='name:' as for some reason this returned weird results
                    # search 'q=' also searches similar e.g. 'martial arts' query also returns results for taekwondo

                    if len(result['playlists']['items']) != 0:

                        for playlist in result['playlists']['items']:
                            playlists.append(playlist['name'])

                        if "playlist" in other_meaningful_info:
                            speak("\nHere's a " + mood_activity_interest + " playlist you might like:\n")
                            for playlist in playlists[len(playlists)-1:]:
                                print(playlist)
                        else:
                            if new_items_count > 1:
                                speak("\n--The results for " + mood_activity_interest + " have been added to the bottom of the previous list.--")

                            speak("\nHere are some " + mood_activity_interest + " playlists you might want to check out:\n")
                            for playlist in playlists:
                                print(playlist)

                    else:
                        speak("\nSorry, I could not find any " + mood_activity_interest + " playlists in the Spotify music library for you.")
                        print("- Tip: For multi-word activities / interests like 'people watching', 'garage sale' and 'martial arts', try switching to text mode, and typing it like this:")
                        print("people-watching, garage-sale, martial-arts")

                except SpotifyException:
                    speak("\nSorry, I was unable to make a search with that mood or activity-interest information.")

            elif "mood" not in entity_links.values() and "activity_interest" not in entity_links.values() and "genre" not in entity_links.values() and len(named_entities_list) == 0:  # for if intent is type of music and no activity, mood or genre is given (e.g. what are some cool songs / albums)
                if new_items_count < 1:  # for case in which no genre, mood or activity/interest is stated, only return results once (e.g. what are some 'cool' and 'fun' songs?)

                    new_items_count += 1

                    try:
                        song_genres = spotifyObject.recommendation_genre_seeds()
                        genre_list = []
                        for name in song_genres['genres']:
                            genre_list.append(name)

                        genre_name = []  # 'seed_genres' needs a list format
                        genre_name.append(genre_list[randint(0, len(genre_list)-1)])

                        result = spotifyObject.recommendations(seed_artists=None, seed_genres=genre_name,
                                                               seed_tracks=None, limit=5, country=None, min_popularity=50, max_popularity=100)
                        if len(result['tracks']) != 0:

                            for track in result['tracks']:
                                for artist_info in track['artists']:
                                    if any(track['name'] in x for x in songs):
                                        index_of_sublist_with_track = next(
                                            (x for x, val in enumerate(songs) if track['name'] in val), None)
                                        songs[index_of_sublist_with_track].insert(
                                            len(songs[index_of_sublist_with_track]) - 1, artist_info['name'])
                                    else:
                                        songs.append([artist_info['name'], track['name']])

                                song_uris.append(track['uri'])

                            if "song" in other_meaningful_info:
                                speak("\nHere's a song by an artist you might like:\n")
                                for song in songs[len(songs)-1:]:
                                    print("1. " + "{} - {}".format(", ".join(song[:-1]), song[-1]))
                                speak("\nIf you'd like to listen to it, just type or say '1'.")
                                print("Note: You can only use this feature if you have Spotify Premium.")
                                print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                            else:
                                speak("\nHere is a selection of songs by artists you might want to check out:\n")
                                for song in songs:
                                    song_name = "{} - {}".format(", ".join(song[:-1]), song[-1])
                                    print(str(songs.index(song) + 1) + ". " + song_name)
                                speak("\nIf you'd like to listen to any of them, just type or say their number (e.g. '1').")
                                print("Note: You can only use this feature if you have premium Spotify account.")
                                print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                        else:
                            speak("\nSorry, I could not find any songs in the Spotify music library for you.\n")

                    except SpotifyException:
                        speak("\nSorry, I was unable to make a search with that information.\n")

    if len(named_entities_list) != 0:

        entity_links = entity_linking_for_named_entities(named_entities_list)
        searched_names = []  # keeping track of searched names so as to not repeat search for e.g. a double search for florence and the machine based on 'florence' and 'the machine' both being in named entities list

        for entity in entity_links:
            if entity_links.get(entity) == "artist":
                search_name = entity  # for some reason 'search' method with type 'artist' cannot use list value, although with type 'genre' needs list value
                artist_name = ""
                artist_uri = ""
                artist_popularity = -1  # for exact matches (cannot be 0 as some artist's popularity is 0)
                artist_popularities = []
                new_items_count += 1

                try:
                    artist_result = spotifyObject.search(q=search_name, limit=15, offset=0, type='artist', market=None)

                    if len(artist_result['artists']['items']) != 0:
                        for artist in artist_result['artists']['items']:
                            if search_name == artist['name'].lower():  # if multiple exact matches, get most popular
                                if artist['popularity'] > artist_popularity:
                                    artist_name = artist['name']
                                    artist_uri = artist['uri']
                                    artist_popularity = artist['popularity']

                            else:  # if no exact name match, get most popular with 'search_name' in their name
                                artist_popularities.append(artist['popularity'])

                        if artist_popularity != -1:
                            if artist_name.lower() not in searched_names:  # for 'exact match' most popular

                                speak("I have used the most popular artist with the name '" + search_name + "' to find results for you.")

                        if artist_popularity == -1:
                            most_popular = max(artist_popularities)
                            for artist in artist_result['artists']['items']:
                                if artist['popularity'] == most_popular:
                                    artist_name = artist['name']
                                    artist_uri = artist['uri']

                            if artist_name.lower() not in searched_names:  # for 'name containing' most popular

                                speak("I have used the artist name '" + artist_name + "' to find results for you.")

                        if artist_name.lower() not in searched_names:
                            searched_names.append(artist_name.lower())

                            if "album" in other_meaningful_info or "albums" in other_meaningful_info:

                                try:
                                    result = spotifyObject.artist_albums(artist_id=artist_uri, album_type=None,
                                                                         country=None, limit=5, offset=randint(0, 5))

                                    if len(result['items']) != 0:

                                        for album in result['items']:
                                            albums.append(album['name'])

                                        if "album" in other_meaningful_info:
                                            speak("\nHere's an album by " + artist_name + " you might like:\n")
                                            for album in albums[len(albums)-1:]:
                                                print(album)
                                        else:
                                            if new_items_count > 1:
                                                speak("\n--The results for " + artist_name + " have been added to the bottom of the previous list.--")

                                            speak("\nHere are some albums by " + artist_name + " you might want to check out:\n")
                                            for album in albums:
                                                print(album)

                                    else:
                                        speak("\nSorry, I could not find any " + artist_name + " albums in the Spotify music library for you.")

                                except SpotifyException:
                                    speak("\nSorry, I was unable to make a search with that artist information.")

                            elif "playlist" in other_meaningful_info or "playlists" in other_meaningful_info:

                                try:
                                    result = spotifyObject.search(q=artist_name, limit=5, offset=randint(0, 10),
                                                                  type='playlist', market=None)

                                    if len(result['playlists']['items']) != 0:

                                        for playlist in result['playlists']['items']:
                                            playlists.append(playlist['name'])

                                        if "playlist" in other_meaningful_info:
                                            speak("\nHere's a playlist for " + artist_name + " you might like:\n")
                                            for playlist in playlists[len(playlists)-1:]:
                                                print(playlist)
                                        else:
                                            if new_items_count > 1:
                                                speak("\n--The results for " + artist_name + " have been added to the bottom of the previous list.--")

                                            speak("\nHere are some playlists for " + artist_name + " you might want to check out:\n")
                                            for playlist in playlists:
                                                print(playlist)

                                    else:
                                        speak("\nSorry, I could not find any " + artist_name + " playlists in the Spotify music library for you.")

                                except SpotifyException:
                                    speak("\nSorry, I was unable to make a search with that artist information.")

                            else:

                                try:
                                    result = spotifyObject.search(q=artist_name, limit=5, offset=randint(0, 15), type='track', market=None)  # for some reason, 'recommendations' method is not working with artist id info

                                    if len(result['tracks']['items']) != 0:

                                        for track in result['tracks']['items']:
                                            for artist_info in track['artists']:
                                                if any(track['name'] in x for x in songs):
                                                    index_of_sublist_with_track = next(
                                                        (x for x, val in enumerate(songs) if track['name'] in val), None)
                                                    songs[index_of_sublist_with_track].insert(
                                                        len(songs[index_of_sublist_with_track]) - 1, artist_info['name'])
                                                else:
                                                    songs.append([artist_info['name'], track['name']])

                                            song_uris.append(track['uri'])

                                        if "song" in other_meaningful_info:
                                            speak("\nHere's a popular song by " + artist_name + " you might like:\n")
                                            for song in songs[len(songs)-1:]:
                                                print("1. " + "{} - {}".format(", ".join(song[:-1]), song[-1]))
                                            speak("\nIf you'd like to listen to it, just type or say '1'.")
                                            print("Note: You can only use this feature if you have Spotify Premium.")
                                            print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                                        else:
                                            if new_items_count > 1:
                                                speak("\n--The results for " + artist_name + " have been added to the bottom of the previous list.--")

                                            speak("\nHere are some popular songs by " + artist_name + " you might want to check out:\n")
                                            for song in songs:
                                                song_name = "{} - {}".format(", ".join(song[:-1]), song[-1])
                                                print(str(songs.index(song) + 1) + ". " + song_name)
                                            speak("\nIf you'd like to listen to any of them, just type or say their number (e.g. '1').")
                                            print("Note: You can only use this feature if you have premium Spotify account.")
                                            print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                                    else:
                                        speak("\nSorry, I could not find any " + artist_name + " songs in the Spotify music library for you.")

                                except SpotifyException:  # if genre not found
                                    speak("\nSorry, I was unable to make a search with that artist information.")

                    else:
                        speak("\nSorry, I could not find any results in the Spotify music library for the artist: " + search_name + ".\n")

                except SpotifyException:
                    speak("\nSorry, I was unable to make a search with that artist information.")

        searched_names.clear()  # re-setting for new input question

    elif len(named_entities_list) == 0 and len(other_meaningful_info) == 0:
        speak("\nSorry, I was unable to make a search based on the information you've given me.")
        print("(Maybe try re-phrasing your sentence and ask me again.)\n")

    return


def similar_to(named_entities_list, other_meaningful_info):
    """
    This method performs search queries for the intent 'similar to'.

    :param named_entities_list: the list of named entities to use for the search query
    :type named_entities_list: list
    :param other_meaningful_info: the list of other meaningful info items to use for the search query
    :type other_meaningful_info: list
    :return none
    """
    albums = []
    songs = []
    artists = []
    song_uris.clear()  # re-setting list for new songs based off new input
    new_items_count = 0  # keeping track of new items in lists

    if len(named_entities_list) != 0:

        entity_links = entity_linking_for_named_entities(named_entities_list)
        searched_names = []

        for entity in entity_links:
            if entity_links.get(entity) == "artist":
                search_name = entity
                artist_name = ""
                artist_uri = ""
                artist_popularity = -1  # for exact matches (cannot be 0 as some artist's popularity is 0)
                artist_popularities = []
                new_items_count += 1

                try:
                    artist_result = spotifyObject.search(q=search_name, limit=15, offset=0, type='artist', market=None)

                    if len(artist_result['artists']['items']) != 0:
                        for artist in artist_result['artists']['items']:
                            if search_name == artist['name'].lower():  # if multiple exact matches, get most popular
                                if artist['popularity'] > artist_popularity:
                                    artist_name = artist['name']
                                    artist_uri = artist['uri']
                                    artist_popularity = artist['popularity']

                            else:  # if no exact name match, get most popular with 'search_name' in their name
                                artist_popularities.append(artist['popularity'])

                        if artist_popularity != -1:
                            if artist_name.lower() not in searched_names:  # for 'exact match' most popular

                                speak("I have used the most popular artist with the name '" + search_name + "' to find results for you.")

                        if artist_popularity == -1:
                            most_popular = max(artist_popularities)
                            for artist in artist_result['artists']['items']:
                                if artist['popularity'] == most_popular:
                                    artist_name = artist['name']
                                    artist_uri = artist['uri']

                            if artist_name.lower() not in searched_names:  # for 'name containing' most popular

                                speak("I have used the artist name '" + artist_name + "' to find results for you.")

                        if artist_name.lower() not in searched_names:
                            searched_names.append(artist_name.lower())

                            try:
                                result = spotifyObject.artist_related_artists(artist_id=artist_uri)

                                if len(result['artists']) != 0:
                                    for artist in result['artists']:
                                        artists.append(artist['name'])

                                    if "artist" in other_meaningful_info or "artists" in other_meaningful_info:

                                        if "artist" in other_meaningful_info:
                                            speak("\nHere's an artist like " + artist_name + " you might like:\n")
                                            for artist in artists[:1]:
                                                print(artist)
                                        else:
                                            speak("\nHere are some artists like " + artist_name + " you might want to check out:\n")
                                            for artist in artists[:5]:  # only want to show maximum 5 similar artists out of the full list retrieved
                                                print(artist)

                                    elif "album" in other_meaningful_info or "albums" in other_meaningful_info:

                                        try:
                                            similar_artist = artists[randint(0, len(artists) - 1)]
                                            similar_artist_result = spotifyObject.search(q=similar_artist, limit=1,
                                                                                         offset=0,
                                                                                         type='artist', market=None)

                                            similar_artist_uri = ""
                                            # have assumed there will be results for this artist as the name was taken from a search result meaning the database contains that artist's info
                                            for artist in similar_artist_result['artists']['items']:
                                                similar_artist_uri = artist['uri']

                                            similar_artist_album = spotifyObject.artist_albums(artist_id=similar_artist_uri, album_type='album', country=None, limit=5, offset=0)  # only getting the albums of 1 similar artist

                                            if similar_artist_album['items'] != 0:
                                                for album in similar_artist_album['items']:
                                                    album_name = album['name']
                                                    for name in album['artists']:  # get names of all artists on that album
                                                        if any(album_name in x for x in albums):
                                                            index_of_sublist_with_album = next(
                                                                (x for x, val in enumerate(albums) if album_name in val),
                                                                None)
                                                            albums[index_of_sublist_with_album].insert(
                                                                len(albums[index_of_sublist_with_album]) - 1,
                                                                name['name'])
                                                        else:
                                                            albums.append([name['name'], album_name])

                                                if "album" in other_meaningful_info:
                                                    speak("\nHere's an album by a similar artist to " + artist_name + " you might like:\n")
                                                    for album in albums[len(albums)-1:]:
                                                        print("{} - {}".format(", ".join(album[:-1]), album[-1]))
                                                else:
                                                    if new_items_count > 1:
                                                        speak("\n--The results for " + artist_name + " have been added to the bottom of the previous list.--")

                                                    speak("\nHere are some albums by a similar artist to " + artist_name + " you might want to check out:\n")
                                                    for album in albums:
                                                        print("{} - {}".format(", ".join(album[:-1]), album[-1]))
                                            else:
                                                speak("\nSorry, I could not find any albums for the similar artist " + artist_name + " in the Spotify music library for you.\n")

                                        except SpotifyException:
                                            speak("\nSorry, I was unable to make a search for albums for the similar artist " + artist_name + " in the Spotify music library for you.\n")

                                    else:
                                        try:
                                            similar_artist = artists[randint(0, len(artists) - 1)]
                                            similar_artist_result = spotifyObject.search(q=similar_artist, limit=1, offset=0,
                                                                                         type='artist', market=None)

                                            similar_artist_name = ""
                                            for artist in similar_artist_result['artists']['items']:
                                                similar_artist_name = artist['name']

                                            similar_artist_music = spotifyObject.search(q=similar_artist_name, limit=5, offset=randint(0, 15), type='track', market=None)

                                            if len(similar_artist_music['tracks']['items']) != 0:

                                                for track in similar_artist_music['tracks']['items']:
                                                    for artist_info in track['artists']:
                                                        if any(track['name'] in x for x in songs):
                                                            index_of_sublist_with_track = next(
                                                                (x for x, val in enumerate(songs) if
                                                                 track['name'] in val), None)
                                                            songs[index_of_sublist_with_track].insert(
                                                                len(songs[index_of_sublist_with_track]) - 1,
                                                                artist_info['name'])
                                                        else:
                                                            songs.append([artist_info['name'], track['name']])

                                                    song_uris.append(track['uri'])

                                                if "song" in other_meaningful_info:
                                                    speak("\nHere's a song by a similar artist to " + artist_name + " you might like:\n")
                                                    for song in songs[len(songs)-1:]:  # read the last entry only
                                                        print("1. " + "{} - {}".format(", ".join(song[:-1]), song[-1]))
                                                    speak("\nIf you'd like to listen to it, just type or say '1'.")
                                                    print("Note: You can only use this feature if you have Spotify Premium.")
                                                    print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                                                else:
                                                    if new_items_count > 1:
                                                        speak("\n--The results for " + artist_name + " have been added to the bottom of the previous list.--")

                                                    speak("\nHere are some songs by a similar artist to " + artist_name + " you might want to check out:\n")
                                                    for song in songs:
                                                        song_name = "{} - {}".format(", ".join(song[:-1]), song[-1])
                                                        print(str(songs.index(song) + 1) + ". " + song_name)
                                                    speak("\nIf you'd like to listen to any of them, just type or say their number (e.g. '1').")
                                                    print("Note: You can only use this feature if you have premium Spotify account.")
                                                    print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                                            else:
                                                speak("\nSorry, I could not find any music for the similar artist " + artist_name + " in the Spotify music library for you.")

                                        except SpotifyException:
                                            speak("\nSorry, I was unable to make a search for music for the similar artist " + artist_name + " in the Spotify music library for you.\n")

                                    artists.clear()

                                else:
                                    speak("\nSorry, I could not find any similar artists to " + search_name + " in the Spotify music library for you.\n")

                            except SpotifyException:
                                speak("\nSorry, I was unable to make a search for a similar artist to " + search_name + " in the Spotify music library for you.\n")

                    else:
                        speak("\nSorry, I could not find any results in the Spotify music library for the artist: " + search_name + ".\n")

                except SpotifyException:

                    speak("\nSorry, I was unable to make a search with that artist information.")

            elif entity_links.get(entity) == "track":
                search_name = entity
                track_name = ""
                track_uri = []  # needs to be list format
                track_popularity = -1
                track_popularities = []
                new_items_count += 1

                try:
                    track_result = spotifyObject.search(q=search_name, limit=15, offset=0, type='track', market=None)

                    if len(track_result['tracks']['items']) != 0:
                        for track in track_result['tracks']['items']:
                            if search_name == track['name'].lower():  # if multiple exact matches, get most popular
                                if track['popularity'] > track_popularity:
                                    track_name = track['name']
                                    track_uri.append(track['uri'])
                                    track_popularity = track['popularity']

                            else:  # if no exact name match, get most popular with 'search_name' in their name
                                track_popularities.append(track['popularity'])

                        if track_popularity != -1:
                            if track_name.lower() not in searched_names:  # for 'exact match' most popular

                                speak("I have used the most popular track with the name '" + search_name + "' to find results for you.")

                        if track_popularity == -1:
                            most_popular = max(track_popularities)
                            for track in track_result['tracks']['items']:
                                if track['popularity'] == most_popular:
                                    track_name = track['name']
                                    track_uri.append(track['uri'])

                            if track_name.lower() not in searched_names:  # for 'name containing' most popular

                                speak("I have used the track name '" + track_name + "' to find results for you.")

                        if track_name.lower() not in searched_names:
                            searched_names.append(track_name.lower())

                            try:
                                result = spotifyObject.recommendations(seed_artists=None, seed_genres=None, seed_tracks=track_uri, limit=5, country=None, min_popularity=50, max_popularity=100)

                                if len(result['tracks']) != 0:

                                    for track in result['tracks']:
                                        for artist_info in track['artists']:
                                            if any(track['name'] in x for x in songs):
                                                index_of_sublist_with_track = next(
                                                    (x for x, val in enumerate(songs) if track['name'] in val), None)
                                                songs[index_of_sublist_with_track].insert(
                                                    len(songs[index_of_sublist_with_track]) - 1, artist_info['name'])
                                            else:
                                                songs.append([artist_info['name'], track['name']])

                                        song_uris.append(track['uri'])

                                    if "song" in other_meaningful_info:
                                        speak("\nHere's a similar song you might like:\n")
                                        for song in songs[len(songs) - 1:]:
                                            print("1. " + "{} - {}".format(", ".join(song[:-1]), song[-1]))
                                        speak("\nIf you'd like to listen to it, just type or say '1'.")
                                        print("Note: You can only use this feature if you have Spotify Premium.")
                                        print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                                    else:
                                        if new_items_count > 1:
                                            speak("\n--The results for " + track_name + " have been added to the bottom of the previous list.--")

                                        speak("\nHere is a selection of similar songs you might want to check out:\n")
                                        for song in songs:
                                            song_name = "{} - {}".format(", ".join(song[:-1]), song[-1])
                                            print(str(songs.index(song) + 1) + ". " + song_name)
                                        speak("\nIf you'd like to listen to any of them, just type or say their number (e.g. '1').")
                                        print("Note: You can only use this feature if you have premium Spotify account.")
                                        print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                                else:
                                    speak("\nSorry, I could not find any similar songs to " + track_name + " in the Spotify music library for you.\n")

                            except SpotifyException:
                                speak("\nSorry, I was unable to make a search for similar tracks to " + track_name + " in the Spotify music library for you.\n")

                    else:
                        speak("\nSorry, I could not find any results for the track " + search_name + " in the Spotify music library for you.\n")

                except SpotifyException:
                    speak("\nSorry, I was unable to make a search with the track " + search_name + ".\n")

            else:

                speak("\nSince the flavour of an album can be quite unique to an artist, it's hard for me to find an album that is really similar to another.")
                speak("Try asking me for albums similar to the style of the specific artist, e.g. What are some albums similar to Kendrick's style?\n")

        searched_names.clear()  # re-setting for new input question

    elif len(named_entities_list) == 0 and len(other_meaningful_info) == 0:
        speak("\nSorry, I was unable to make a search based on the information you've given me.")
        print("(Maybe try re-phrasing your sentence and ask me again.)")

    return


def popularity(named_entities_list, other_meaningful_info):
    """
    This method performs search queries for the intent 'popularity'.

    :param named_entities_list: the list of named entities to use for the search query
    :type named_entities_list: list
    :param other_meaningful_info: the list of other meaningful info items to use for the search query
    :type other_meaningful_info: list
    :return none
    """
    albums_main = []
    songs = []
    song_uris.clear()  # resetting list for new songs based off new input
    new_items_count = 0  # keeping track of new items in lists

    if len(named_entities_list) != 0:

        entity_links = entity_linking_for_named_entities(named_entities_list)
        searched_names = []

        for entity in entity_links:
            if entity_links.get(entity) == "artist":
                search_name = entity
                artist_name = ""
                artist_uri = ""
                artist_popularity = -1
                artist_popularities = []
                new_items_count += 1

                try:
                    artist_result = spotifyObject.search(q=search_name, limit=15, offset=0, type='artist', market=None)

                    if len(artist_result['artists']['items']) != 0:
                        for artist in artist_result['artists']['items']:
                            if search_name == artist['name'].lower():  # if multiple exact matches, get most popular
                                if artist['popularity'] > artist_popularity:
                                    artist_name = artist['name']
                                    artist_uri = artist['uri']
                                    artist_popularity = artist['popularity']

                            else:  # if no exact name match, get most popular with 'search_name' in their name
                                artist_popularities.append(artist['popularity'])

                        if artist_popularity != -1:
                            if artist_name.lower() not in searched_names:  # for 'exact match' most popular

                                speak("I have used the most popular artist with the name '" + search_name + "' to find results for you.")

                        if artist_popularity == -1:
                            most_popular = max(artist_popularities)
                            for artist in artist_result['artists']['items']:
                                if artist['popularity'] == most_popular:
                                    artist_name = artist['name']
                                    artist_uri = artist['uri']

                            if artist_name.lower() not in searched_names:  # for 'name containing' most popular

                                speak("I have used the artist name '" + artist_name + "' to find results for you.")

                        if artist_name.lower() not in searched_names:
                            searched_names.append(artist_name.lower())

                            if "album" in other_meaningful_info or "albums" in other_meaningful_info or "album" in entity_links.values():

                                try:
                                    artist_album = spotifyObject.artist_albums(artist_id=artist_uri,
                                                                                       album_type='album', country=None,
                                                                                       limit=None,
                                                                                       offset=0)

                                    if len(artist_album['items']) != 0:

                                        if "album" in entity_links.values():

                                            album_uri = ""
                                            search_albums = []
                                            album_name = ""

                                            for item, label in entity_links.items():
                                                if label == "album":
                                                    search_albums.append(item)

                                            for item in search_albums:

                                                found = False
                                                for album in artist_album['items']:
                                                    if album['name'].lower() == item.lower():
                                                        album_uri = album['uri']
                                                        album_name = album['name']
                                                        found = True
                                                if not found:
                                                    speak("\nSorry, I was unable to find the album " + item + " in " + artist_name + "'s list of albums.\n")

                                                if album_uri != "":
                                                    album_tracks = spotifyObject.album_tracks(album_id=album_uri, limit=None, offset=None)

                                                    track_uris = []
                                                    most_popular_track = ""
                                                    track_popularities = []
                                                    tracks_list = []

                                                    for track in album_tracks['items']:
                                                        track_uris.append(track['uri'])

                                                    tracks = spotifyObject.tracks(tracks=track_uris)

                                                    for track in tracks['tracks']:
                                                        track_popularities.append(track['popularity'])
                                                        tracks_list.append(track)

                                                    track_popularities.sort(reverse=True)  # sort in descending order of popularity (100 is highest)
                                                    track_popularities = track_popularities[:5]  # only keeping 5 of the tracks from the album (i.e. the 5 most popular)
                                                    max_popularity = max(track_popularities)

                                                    tracks_new_list = []  # for new list of 5 most popular tracks

                                                    for track in tracks_list:
                                                        if track['popularity'] in track_popularities and len(tracks_new_list) < 5:
                                                            tracks_new_list.append(track)
                                                        if track['popularity'] == max_popularity:
                                                            most_popular_track = track['name']

                                                    for track in tracks_new_list:
                                                        for artist_info in track['artists']:
                                                            if any(track['name'] in x for x in songs):
                                                                index_of_sublist_with_track = next(
                                                                    (x for x, val in enumerate(songs) if
                                                                     track['name'] in val), None)
                                                                songs[index_of_sublist_with_track].insert(
                                                                    len(songs[index_of_sublist_with_track]) - 1,
                                                                    artist_info['name'])
                                                            else:
                                                                songs.append([artist_info['name'], track['name']])

                                                        song_uris.append(track['uri'])

                                                    if "song" in other_meaningful_info:
                                                        speak("\nHere is the most popular song on " + artist_name + "'s " + album_name + " album:\n")
                                                        print("1. " + artist_name + " - " + most_popular_track)
                                                        speak("\nIf you'd like to listen to it, just type or say '1'.")
                                                        print("Note: You can only use this feature if you have Spotify Premium.")
                                                        print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                                                    else:
                                                        if new_items_count > 1:
                                                            speak("\n--The results for " + album_name + " have been added to the bottom of the previous list.--")

                                                        speak("\nHere are the most popular songs on " + artist_name + "'s " + album_name + " album:\n")
                                                        for song in songs:
                                                            song_name = "{} - {}".format(", ".join(song[:-1]), song[-1])
                                                            print(str(songs.index(song) + 1) + ". " + song_name)
                                                        speak("\nIf you'd like to listen to any of them, just type or say their number (e.g. '1').")
                                                        print("Note: You can only use this feature if you have premium Spotify account.")
                                                        print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                                                new_items_count += 1

                                        else:

                                            most_popular_album = ""
                                            album_uris = []
                                            album_popularities = []
                                            albums_list = []

                                            for album in artist_album['items']:
                                                album_uris.append(album['uri'])

                                            albums = spotifyObject.albums(albums=album_uris)

                                            for album in albums['albums']:
                                                album_popularities.append(album['popularity'])
                                                albums_list.append(album)

                                            album_popularities.sort(reverse=True)  # sort in descending order of popularity (100 is highest)
                                            album_popularities = album_popularities[:5]  # only keeping 5 of the albums (i.e. the 5 most popular)
                                            max_popularity = max(album_popularities)

                                            albums_new_list = []  # for new list of 5 most popular albums

                                            for album in albums_list:
                                                if album['popularity'] in album_popularities and len(albums_new_list) < 5:
                                                    albums_new_list.append(album)
                                                if album['popularity'] == max_popularity:
                                                    most_popular_album = album['name']

                                            for album in albums_new_list:
                                                for name in album['artists']:  # get names of all artists on that album
                                                    if any(album['name'] in x for x in albums_main):
                                                        index_of_sublist_with_album = next(
                                                            (x for x, val in enumerate(albums_main) if album['name'] in val),
                                                            None)
                                                        albums_main[index_of_sublist_with_album].insert(
                                                            len(albums_main[index_of_sublist_with_album]) - 1, name['name'])
                                                    else:
                                                        albums_main.append([name['name'], album['name']])

                                            if "album" in other_meaningful_info:
                                                speak("\nHere's " + artist_name + "'s most popular album:\n")
                                                print(artist_name + " - " + most_popular_album)
                                            else:
                                                if new_items_count > 1:
                                                    speak("\n--The results for " + artist_name + " have been added to the bottom of the previous list.--")

                                                speak("\nHere are " + artist_name + "'s most popular albums:\n")
                                                for album in albums_main:
                                                    print("{} - {}".format(", ".join(album[:-1]), album[-1]))
                                    else:
                                        speak("\nSorry, I could not find any top albums for " + artist_name + " in the Spotify music library for you.\n")

                                except SpotifyException:
                                    speak("\nSorry, I was unable to make a search for the most popular albums by " + artist_name + ".\n")

                            else:

                                try:
                                    result = spotifyObject.artist_top_tracks(artist_id=artist_uri, country='GB')  # country required for 'top_tracks' method to work

                                    if len(result['tracks']) != 0:

                                        most_popular_track = ""
                                        track_popularity = 0

                                        for track in result['tracks'][:5]:
                                            if track['popularity'] > track_popularity:
                                                track_popularity = track['popularity']
                                                most_popular_track = track['name']
                                            for artist_info in track['artists']:
                                                if any(track['name'] in x for x in songs):
                                                    index_of_sublist_with_track = next(
                                                        (x for x, val in enumerate(songs) if track['name'] in val), None)
                                                    songs[index_of_sublist_with_track].insert(
                                                        len(songs[index_of_sublist_with_track]) - 1, artist_info['name'])
                                                else:
                                                    songs.append([artist_info['name'], track['name']])

                                            song_uris.append(track['uri'])

                                        if "song" in other_meaningful_info:
                                            speak("\nHere's " + artist_name + "'s most popular song:\n")
                                            print("1. " + artist_name + " - " + most_popular_track)
                                            speak("\nIf you'd like to listen to it, just type or say '1'.")
                                            print("Note: You can only use this feature if you have Spotify Premium.")
                                            print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                                        else:
                                            if new_items_count > 1:
                                                speak("\n--The results for " + artist_name + " have been added to the bottom of the previous list.--")

                                            speak("\nHere are " + artist_name + "'s most popular songs:\n")
                                            for song in songs:
                                                song_name = "{} - {}".format(", ".join(song[:-1]), song[-1])
                                                print(str(songs.index(song) + 1) + ". " + song_name)
                                            speak("\nIf you'd like to listen to any of them, just type or say their number (e.g. '1').")
                                            print("Note: You can only use this feature if you have premium Spotify account.")
                                            print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                                    else:
                                        speak("\nSorry, I could not find any top tracks for " + artist_name + " in the Spotify music library for you.\n")

                                except SpotifyException:
                                    speak("\nSorry, I was unable to make a search for the most popular tracks by " + artist_name + ".\n")
                    else:
                        speak("\nSorry, I could not find any results in the Spotify music library for the artist: " + search_name + ".\n")

                except SpotifyException:
                    speak("\nSorry, I was unable to make a search with that artist information.")

            elif entity_links.get(entity) == "album" and "artist" not in entity_links.values():
                search_name = entity
                album_name = ""
                album_uri = ""
                album_popularity = -1
                album_popularities = []
                new_items_count += 1

                try:
                    album_result = spotifyObject.search(q=search_name, limit=15, offset=0, type='album', market=None)

                    if len(album_result['albums']['items']) != 0:
                        for album in album_result['albums']['items']:
                            album_full = spotifyObject.album(album_id=album['uri'])  # need to get full album object to get 'popularity'
                            if search_name == album['name'].lower():  # if multiple exact matches, get most popular
                                if album_full['popularity'] > album_popularity:
                                    album_name = album['name']
                                    album_uri = album['uri']
                                    album_popularity = album_full['popularity']

                            else:  # if no exact name match, get most popular with 'search_name' in their name
                                album_popularities.append(album_full['popularity'])

                        if album_popularity != -1:
                            if album_name.lower() not in searched_names:  # for 'exact match' most popular

                                speak("I have used the most popular album with the name '" + search_name + "' to find results for you.")

                        if album_popularity == -1:
                            most_popular = max(album_popularities)
                            for album in album_result['albums']['items']:
                                album_full = spotifyObject.album(album_id=album['uri'])  # need to get full album object to get 'popularity'
                                if album_full['popularity'] == most_popular:
                                    album_name = album['name']
                                    album_uri = album['uri']

                            if album_name.lower() not in searched_names:  # for 'name containing' most popular

                                speak("I have used the album name '" + album_name + "' to find results for you.")

                        if album_name.lower() not in searched_names:
                            searched_names.append(album_name.lower())

                            album_tracks = spotifyObject.album_tracks(album_id=album_uri, limit=None,
                                                                      offset=None)

                            track_uris = []
                            most_popular_track = ""
                            track_popularities = []
                            tracks_list = []

                            for track in album_tracks['items']:
                                track_uris.append(track['uri'])

                            tracks = spotifyObject.tracks(tracks=track_uris)

                            for track in tracks['tracks']:
                                track_popularities.append(track['popularity'])
                                tracks_list.append(track)

                            track_popularities.sort(reverse=True)  # sort in descending order of popularity (100 is highest)
                            track_popularities = track_popularities[:5]  # only keeping 5 of the tracks from the album (i.e. the 5 most popular)
                            max_popularity = max(track_popularities)

                            tracks_new_list = []  # for new list of 5 most popular tracks

                            for track in tracks_list:
                                if track['popularity'] in track_popularities and len(tracks_new_list) < 5:
                                    tracks_new_list.append(track)
                                if track['popularity'] == max_popularity:
                                    most_popular_track = track['name']

                            for track in tracks_new_list:
                                for artist_info in track['artists']:
                                    if any(track['name'] in x for x in songs):
                                        index_of_sublist_with_track = next(
                                            (x for x, val in enumerate(songs) if
                                             track['name'] in val), None)
                                        songs[index_of_sublist_with_track].insert(
                                            len(songs[index_of_sublist_with_track]) - 1,
                                            artist_info['name'])
                                    else:
                                        songs.append([artist_info['name'], track['name']])

                                song_uris.append(track['uri'])

                            if "song" in other_meaningful_info:
                                speak("\nHere is the most popular song on the album " + album_name + ":\n")
                                print("1. " + most_popular_track)
                                speak("\nIf you'd like to listen to it, just type or say '1'.")
                                print("Note: You can only use this feature if you have Spotify Premium.")
                                print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                            else:
                                if new_items_count > 1:
                                    speak("\n--The results for " + album_name + " have been added to the bottom of the previous list.--")

                                speak("\nHere are the most popular songs on the album " + album_name + ":\n")
                                for song in songs:
                                    song_name = "{} - {}".format(", ".join(song[:-1]), song[-1])
                                    print(str(songs.index(song) + 1) + ". " + song_name)
                                speak("\nIf you'd like to listen to any of them, just type or say their number (e.g. '1').")
                                print("Note: You can only use this feature if you have premium Spotify account.")
                                print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                            # new_items_count += 1

                    else:
                        speak("\nSorry, I could not find any results for the album " + search_name + " in the Spotify music library for you.\n")

                except SpotifyException:
                    speak("\nSorry, I was unable to make a search for the album " + search_name + ".\n")

        searched_names.clear()  # re-setting for new input question

    elif len(other_meaningful_info) != 0 and len(named_entities_list) == 0:

        try:
            global_top_50 = "https://api.spotify.com/v1/playlists/37i9dQZEVXbMDoHDwVN2tF"

            result = spotifyObject.user_playlist_tracks(user="spotifycharts", playlist_id=global_top_50, fields=None, limit=5, offset=0, market=None)

            if len(result['items']) != 0:

                for track in result['items']:
                    for artist_info in track['track']['artists']:
                        if any(track['track']['name'] in x for x in songs):
                            index_of_sublist_with_track = next(
                                (x for x, val in enumerate(songs) if
                                 track['track']['name'] in val), None)
                            songs[index_of_sublist_with_track].insert(
                                len(songs[index_of_sublist_with_track]) - 1,
                                artist_info['name'])
                        else:
                            songs.append([artist_info['name'], track['track']['name']])

                    song_uris.append(track['track']['uri'])

                if "song" in other_meaningful_info or "artist" in other_meaningful_info:
                    speak("\nHere is the number one song and artist currently in Global Top 50:\n")
                    for song in songs[:1]:
                        print("1. " + "{} - {}".format(", ".join(song[:-1]), song[-1]))
                    speak("\nIf you'd like to listen to it, just type or say '1'.")
                    print("Note: You can only use this feature if you have Spotify Premium.")
                    print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

                elif "album" in other_meaningful_info or "albums" in other_meaningful_info:
                    speak("\nSorry, I do not have any information on the top albums currently in the Spotify music library for you.")
                    speak("Try asking for the most popular songs right now, or the most popular albums of a particular artist.")

                else:
                    speak("\nHere are the 5 most popular songs and artists currently in Global Top 50:\n")
                    for song in songs:
                        song_name = "{} - {}".format(", ".join(song[:-1]), song[-1])
                        print(str(songs.index(song) + 1) + ". " + song_name)
                    speak("\nIf you'd like to listen to any of them, just type or say their number (e.g. '1').")
                    print("Note: You can only use this feature if you have premium Spotify account.")
                    print("You'll also need to have Spotify Web or Spotify Desktop open for me to play the song in.")

            else:
                speak("\nSorry, I could not find any results for Global Top 50 in the Spotify music library for you.\n")

        except SpotifyException:
            speak("\nSorry, I was unable to make a search for the Global Top 50.\n")

    elif len(named_entities_list) == 0 and len(other_meaningful_info) == 0:
        speak("\nSorry, I was unable to make a search based on the information you've given me.")
        print("(Maybe try re-phrasing your sentence and ask me again.)")

    return


def artist_info(named_entities_list):
    """
    This method performs search queries for the intent 'artist info'.

    :param named_entities_list: the list of named entities to use for the search query
    :type named_entities_list: list
    :return none
    """
    artists = []

    if len(named_entities_list) != 0:

        entity_links = entity_linking_for_named_entities(named_entities_list)
        searched_names = []

        for entity in entity_links:
            if entity_links.get(entity) == "track":
                search_name = entity
                track_name = ""
                track_popularity = -1
                track_popularities = []

                try:
                    track_result = spotifyObject.search(q=search_name, limit=15, offset=0, type='track', market=None)

                    if len(track_result['tracks']['items']) != 0:
                        for track in track_result['tracks']['items']:
                            if search_name == track['name'].lower():  # if multiple exact matches, get most popular
                                if track['popularity'] > track_popularity:
                                    track_name = track['name']
                                    track_popularity = track['popularity']
                                    artists.clear()
                                    for artist in track['artists']:
                                        artists.append(artist['name'])

                            else:  # if no exact name match, get most popular with 'search_name' in their name
                                track_popularities.append(track['popularity'])

                        if track_popularity != -1:
                            if track_name.lower() not in searched_names:  # for 'exact match' most popular

                                speak("I have used the most popular track with the name '" + search_name + "' to find results for you.")

                        elif track_popularity == -1:
                            most_popular = max(track_popularities)
                            for track in track_result['tracks']['items']:
                                if track['popularity'] == most_popular:
                                    track_name = track['name']
                                    artists.clear()
                                    for artist in track['artists']:
                                        artists.append(artist['name'])

                            if track_name.lower() not in searched_names:  # for 'name containing' most popular

                                speak("I have used the track name '" + track_name + "' to find results for you.")

                        if track_name.lower() not in searched_names:
                            searched_names.append(track_name.lower())

                            if len(artists) == 1:
                                speak("\nThe song is by the artist: " + artists[0])
                            else:
                                speak("\nThe song is by these artists:")
                                for artist in artists:
                                    print(artist)

                        artists.clear()  # reset list for new track search

                    else:
                        speak("\nSorry, I could not find any results for the track " + search_name + " in the Spotify music library for you.\n")

                except SpotifyException:
                    speak("\nSorry, I was unable to make a search with the track " + search_name + ".\n")

            elif entity_links.get(entity) == "album":
                search_name = entity
                album_name = ""
                album_popularity = -1
                album_popularities = []

                try:
                    album_result = spotifyObject.search(q=search_name, limit=15, offset=0, type='album', market=None)

                    if len(album_result['albums']['items']) != 0:
                        for album in album_result['albums']['items']:
                            album_full = spotifyObject.album(album_id=album['uri'])  # need to get full album object to get 'popularity'
                            if search_name == album['name'].lower():  # if multiple exact matches, get most popular
                                if album_full['popularity'] > album_popularity:
                                    album_name = album['name']
                                    album_popularity = album_full['popularity']
                                    artists.clear()
                                    for artist in album['artists']:
                                        artists.append(artist['name'])

                            else:  # if no exact name match, get most popular with 'search_name' in their name
                                album_popularities.append(album_full['popularity'])

                        if album_popularity != -1:
                            if album_name.lower() not in searched_names:  # for 'exact match' most popular

                                speak("I have used the most popular album with the name '" + search_name + "' to find results for you.")

                        elif album_popularity == -1:
                            most_popular = max(album_popularities)
                            for album in album_result['albums']['items']:
                                album_full = spotifyObject.album(album_id=album['uri'])  # need to get full album object to get 'popularity'
                                if album_full['popularity'] == most_popular:
                                    album_name = album['name']
                                    artists.clear()
                                    for artist in album['artists']:
                                        artists.append(artist['name'])

                            if album_name.lower() not in searched_names:  # for 'name containing' most popular

                                speak("I have used the album name '" + album_name + "' to find results for you.")

                        if album_name.lower() not in searched_names:
                            searched_names.append(album_name.lower())

                            if len(artists) == 1:
                                speak("\nThe album is by the artist: " + artists[0])
                            else:
                                speak("\nThe album is by these artists:")
                                for artist in artists:
                                    print(artist)

                        artists.clear()  # reset list for new track search

                    else:
                        speak("\nSorry, I could not find any results for the album " + search_name + " in the Spotify music library for you.\n")

                except SpotifyException:
                    speak("\nSorry, I was unable to make a search with the album " + search_name + ".\n")

    else:
        speak("\nSorry, I was unable to make a search based on the information you've given me.")
        print("(Maybe try re-phrasing your sentence and ask me again.)")

    return


def album_info(named_entities_list):
    """
    This method performs search queries for the intent 'album info'.

    :param named_entities_list: the list of named entities to use for the search query
    :type named_entities_list: list
    :return none
    """
    album_name = ""

    if len(named_entities_list) != 0:

        entity_links = entity_linking_for_named_entities(named_entities_list)
        searched_names = []

        for entity in entity_links:

            if entity_links.get(entity) == "track":
                search_name = entity
                track_name = ""
                track_popularity = -1
                track_popularities = []

                try:
                    track_result = spotifyObject.search(q=search_name, limit=15, offset=0, type='track', market=None)

                    if len(track_result['tracks']['items']) != 0:
                        found = False  # keeping track of if an accurate match has been found
                        for track in track_result['tracks']['items']:
                            if not found:
                                if search_name == track['name'].lower():  # if multiple exact matches, get most popular
                                    if "artist" in entity_links.values():
                                        artist_name = ""
                                        for item in entity_links:
                                            if entity_links.get(item) == "artist":
                                                artist_name = item
                                        for artist in track['artists']:
                                            name = artist['name']
                                            if artist_name in name.lower():  # all entity links already lowercase i.e. 'artist_name'
                                                track_name = track['name']
                                                album_name = track['album']['name']
                                                found = True
                                                break

                                            else:
                                                if track['popularity'] > track_popularity:
                                                    track_name = track['name']
                                                    track_popularity = track['popularity']
                                                    album_name = track['album']['name']
                                    else:
                                        if track['popularity'] > track_popularity:
                                            track_name = track['name']
                                            track_popularity = track['popularity']
                                            album_name = track['album']['name']

                                else:  # if no exact name match, get most popular with 'search_name' in their name
                                    if "artist" in entity_links.values():
                                        artist_name = ""
                                        for item in entity_links:
                                            if entity_links.get(item) == "artist":
                                                artist_name = item
                                        for artist in track['artists']:
                                            name = artist['name']
                                            if artist_name in name.lower():
                                                track_name = track['name']
                                                album_name = track['album']['name']
                                                found = True
                                                break

                                            else:
                                                track_popularities.append(track['popularity'])
                                    else:
                                        track_popularities.append(track['popularity'])

                        if track_popularity != -1 or track_name != "":
                            if track_name.lower() not in searched_names:  # for 'exact match' most popular

                                speak("I have used the most popular track with the name '" + search_name + "' to find results for you.")

                        elif track_popularity == -1 and track_name == "":
                            most_popular = max(track_popularities)
                            for track in track_result['tracks']['items']:
                                if track['popularity'] == most_popular:
                                    track_name = track['name']
                                    album_name = track['album']['name']

                            if track_name.lower() not in searched_names:  # for 'name containing' most popular

                                speak("I have used the track name '" + track_name + "' to find results for you.")

                        if track_name.lower() not in searched_names:
                            searched_names.append(track_name.lower())

                            speak("\nThe song is on the album: " + album_name)

                    else:
                        speak("\nSorry, I could not find any results for the track " + search_name + " in the Spotify music library for you.\n")

                except SpotifyException:
                    speak("\nSorry, I was unable to make a search with the track " + search_name + ".\n")

    else:
        speak("\nSorry, I was unable to make a search based on the information you've given me.")
        print("(Maybe try re-phrasing your sentence and ask me again.)")

    return


def track_info(named_entities_list, other_meaningful_info):
    """
    This method performs search queries for the intent 'track info'.

    :param named_entities_list: the list of named entities to use for the search query
    :type named_entities_list: list
    :param other_meaningful_info: the list of other meaningful info items to use for the search query
    :type other_meaningful_info: list
    :return none
    """
    tracks = []
    new_items_count = 0

    if len(named_entities_list) != 0:

        entity_links = entity_linking_for_named_entities(named_entities_list)
        searched_names = []

        for entity in entity_links:

            if entity_links.get(entity) == "album":
                search_name = entity
                album_name = ""
                album_popularity = -1
                album_popularities = []
                new_items_count += 1

                try:
                    album_result = spotifyObject.search(q=search_name, limit=15, offset=0, type='album', market=None)

                    if len(album_result['albums']['items']) != 0:
                        found = False  # keeping track of if an accurate match has been found

                        for album in album_result['albums']['items']:
                            if not found:
                                if search_name == album['name'].lower():  # if multiple exact matches, get most popular
                                    if "artist" in entity_links.values():
                                        artist_name = ""
                                        for item in entity_links:
                                            if entity_links.get(item) == "artist":
                                                artist_name = item
                                        for artist in album['artists']:
                                            name = artist['name']
                                            if artist_name in name.lower():  # all entity links already lowercase i.e. 'artist_name'
                                                album_name = album['name']
                                                album_tracks = spotifyObject.album_tracks(album_id=album['uri'], limit=None,
                                                                                          offset=None)
                                                if new_items_count > 1:
                                                    tracks.clear()
                                                for track in album_tracks['items']:
                                                    if track['name'] not in tracks:
                                                        tracks.append(track['name'])
                                                found = True
                                                break

                                            else:
                                                album_full = spotifyObject.album(album_id=album[
                                                    'uri'])  # need to get full album object to get 'popularity'
                                                if album_full['popularity'] > album_popularity:
                                                    album_name = album['name']
                                                    album_popularity = album_full['popularity']
                                                    album_tracks = spotifyObject.album_tracks(album_id=album['uri'],
                                                                                              limit=None,
                                                                                              offset=None)
                                                    tracks.clear()  # needs to be cleared each time as it's getting replaced with tracks of the more popular album each time in the for loop
                                                    for track in album_tracks['items']:
                                                        if track['name'] not in tracks:
                                                            tracks.append(track['name'])

                                    else:
                                        album_full = spotifyObject.album(album_id=album['uri'])  # need to get full album object to get 'popularity'
                                        if album_full['popularity'] > album_popularity:
                                            album_name = album['name']
                                            album_popularity = album_full['popularity']
                                            album_tracks = spotifyObject.album_tracks(album_id=album['uri'], limit=None,
                                                                                      offset=None)
                                            tracks.clear()
                                            for track in album_tracks['items']:
                                                if track['name'] not in tracks:
                                                    tracks.append(track['name'])

                                else:  # if no exact name match, get most popular with 'search_name' in their name
                                    if "artist" in entity_links.values():
                                        artist_name = ""
                                        for item in entity_links:
                                            if entity_links.get(item) == "artist":
                                                artist_name = item
                                        for artist in album['artists']:
                                            name = artist['name']
                                            if artist_name in name.lower():
                                                album_name = album['name']
                                                album_tracks = spotifyObject.album_tracks(album_id=album['uri'], limit=None,
                                                                                          offset=None)
                                                if new_items_count > 1:
                                                    tracks.clear()
                                                for track in album_tracks['items']:
                                                    if track['name'] not in tracks:
                                                        tracks.append(track['name'])
                                                found = True
                                                break

                                            else:
                                                album_full = spotifyObject.album(album_id=album[
                                                    'uri'])  # need to get full album object to get 'popularity'
                                                if album_full['popularity'] > album_popularity:
                                                    album_name = album['name']
                                                    album_popularities.append(album_full['popularity'])

                                    else:
                                        album_full = spotifyObject.album(album_id=album['uri'])  # need to get full album object to get 'popularity'
                                        if album_full['popularity'] > album_popularity:
                                            album_name = album['name']
                                            album_popularities.append(album_full['popularity'])

                        if album_popularity != -1 or album_name != "":
                            if album_name.lower() not in searched_names:  # for 'exact match' most popular

                                speak("I have used the most popular album with the name '" + search_name + "' to find results for you.")

                        elif album_popularity == -1 and album_name == "":
                            most_popular = max(album_popularities)
                            for album in album_result['albums']['items']:
                                album_full = spotifyObject.album(album_id=album['uri'])
                                if album_full['popularity'] == most_popular:
                                    album_name = album['name']
                                    album_tracks = spotifyObject.album_tracks(album_id=album['uri'], limit=None,
                                                                              offset=None)

                                    if new_items_count > 1:
                                        tracks.clear()
                                    for track in album_tracks['items']:
                                        if track['name'] not in tracks:
                                            tracks.append(track['name'])

                            if album_name.lower() not in searched_names:  # for 'name containing' most popular

                                speak("I have used the album name '" + album_name + "' to find results for you.")

                        if album_name.lower() not in searched_names:
                            searched_names.append(album_name.lower())

                            if "song" in other_meaningful_info:
                                speak("\nHere is a song from the album " + album_name + ":\n")
                                print(tracks[0])

                            else:
                                speak("\nHere are the songs on the album " + album_name + ":\n")
                                for track in tracks:
                                    print(track)

                    else:
                        speak("\nSorry, I could not find any results for the album " + search_name + " in the Spotify music library for you.\n")

                except SpotifyException:
                    speak("\nSorry, I was unable to make a search with the album " + search_name + ".\n")

    else:
        speak("\nSorry, I was unable to make a search based on the information you've given me.")
        print("(Maybe try re-phrasing your sentence and ask me again.)")

    return


def play(named_entities_list):
    """
    This method performs search queries for the intent 'play'.

    :param named_entities_list: the list of named entities to use for the search query
    :type named_entities_list: list
    :return none
    """
    artist_name = ""
    song_uri = []
    count = 0

    if len(named_entities_list) != 0:

        entity_links = entity_linking_for_named_entities(named_entities_list)
        searched_names = []

        if "track" in entity_links.values():

            for entity in entity_links:

                if entity_links.get(entity) == "track" and count < 1:  # only play song
                    search_name = entity
                    track_name = ""
                    track_popularity = -1
                    track_popularities = []
                    count += 1

                    try:
                        track_result = spotifyObject.search(q=search_name, limit=15, offset=0, type='track', market=None)

                        if len(track_result['tracks']['items']) != 0:
                            found = False  # keeping track of if an accurate match has been found
                            for track in track_result['tracks']['items']:
                                if not found:
                                    if search_name == track['name'].lower():  # if multiple exact matches, get most popular
                                        if "artist" in entity_links.values():
                                            artist = ""
                                            for item in entity_links:
                                                if entity_links.get(item) == "artist":
                                                    artist = item
                                            name = track['artists'][0]['name']
                                            if artist in name.lower():  # all entity links already lowercase i.e. 'artist_name'
                                                track_name = track['name']
                                                artist_name = track['artists'][0]['name']
                                                song_uri.append(track['uri'])
                                                found = True
                                                break  # if name found, no need to continue loop

                                            else:
                                                if track['popularity'] > track_popularity:
                                                    track_name = track['name']
                                                    track_popularity = track['popularity']
                                                    artist_name = track['artists'][0]['name']
                                                    song_uri.append(track['uri'])
                                        else:
                                            if track['popularity'] > track_popularity:
                                                track_name = track['name']
                                                track_popularity = track['popularity']
                                                artist_name = track['artists'][0]['name']
                                                song_uri.append(track['uri'])

                                    else:  # if no exact name match, get most popular with 'search_name' in their name
                                        if "artist" in entity_links.values():
                                            artist = ""
                                            for item in entity_links:
                                                if entity_links.get(item) == "artist":
                                                    artist = item
                                            name = track['artists'][0]['name']
                                            if artist in name.lower():  # all entity links already lowercase i.e. 'artist_name'
                                                track_name = track['name']
                                                artist_name = track['artists'][0]['name']
                                                song_uri.append(track['uri'])
                                                found = True
                                                break

                                            else:
                                                track_popularities.append(track['popularity'])
                                        else:
                                            track_popularities.append(track['popularity'])

                            if track_popularity != -1 or track_name != "":
                                if track_name.lower() not in searched_names:  # for 'exact match' most popular

                                    speak("\nI have used the most popular track with the name '" + search_name + "' to find results for you.")

                            elif track_popularity == -1 and track_name == "":
                                most_popular = max(track_popularities)
                                for track in track_result['tracks']['items']:
                                    if track['popularity'] == most_popular:
                                        track_name = track['name']
                                        artist_name = track['artists'][0]['name']
                                        song_uri.append(track['uri'])

                                if track_name.lower() not in searched_names:  # for 'name containing' most popular

                                    speak("\nI have used the track name '" + track_name + "' to find results for you.")

                            if track_name.lower() not in searched_names:
                                searched_names.append(track_name.lower())

                                try:
                                    speak("\nPlaying: " + track_name + " by " + artist_name)
                                    spotifyObject.start_playback(device_id=None, context_uri=None, uris=song_uri, offset=None)  # requires list format for 'uris'

                                except SpotifyException:
                                    speak("\nSorry, you'll need Spotify Premium to play the songs.")
                                    speak("\nIf you have Premium, make sure you've got a Spotify player open for me to utilise - such as Spotify Web or Spotify Desktop.")
                                    print("- if you have one of those open, try playing and pausing any track to test it's all working, then type the number of the song you want again.")

                        else:
                            speak("\nSorry, I could not find any results for the track " + search_name + " in the Spotify music library for you.\n")

                    except SpotifyException:
                        speak("\nSorry, I was unable to make a search with the track " + search_name + ".\n")

        elif "artist" in entity_links.values() and "track" not in entity_links.values():

            for entity in entity_links:

                if entity_links.get(entity) == "artist" and count < 1:
                    search_name = entity
                    artist_popularity = -1
                    artist_popularities = []
                    track_name = ""
                    count += 1

                    try:
                        artist_result = spotifyObject.search(q=search_name, limit=15, offset=0, type='artist',
                                                             market=None)

                        if len(artist_result['artists']['items']) != 0:
                            for artist in artist_result['artists']['items']:
                                if search_name == artist['name'].lower():  # if multiple exact matches, get most popular
                                    if artist['popularity'] > artist_popularity:
                                        artist_name = artist['name']
                                        artist_popularity = artist['popularity']

                                else:  # if no exact name match, get most popular with 'search_name' in their name
                                    artist_popularities.append(artist['popularity'])

                            if artist_popularity != -1:
                                if artist_name.lower() not in searched_names:  # for 'exact match' most popular

                                    speak("I have used the most popular artist with the name '" + search_name + "' to find results for you.")

                            if artist_popularity == -1:
                                most_popular = max(artist_popularities)
                                for artist in artist_result['artists']['items']:
                                    if artist['popularity'] == most_popular:
                                        artist_name = artist['name']

                                if artist_name.lower() not in searched_names:  # for 'name containing' most popular

                                    speak("I have used the artist name '" + artist_name + "' to find results for you.")

                            if artist_name.lower() not in searched_names:
                                searched_names.append(artist_name.lower())

                                try:
                                    track_result = spotifyObject.search(q=artist_name, limit=1, offset=randint(0, 15), type='track', market=None)

                                    if len(track_result['tracks']['items']) != 0:
                                        for track in track_result['tracks']['items']:
                                            track_name = track['name']
                                            song_uri.append(track['uri'])

                                        try:
                                            speak("\nPlaying: " + track_name + " by " + artist_name)
                                            spotifyObject.start_playback(device_id=None, context_uri=None,
                                                                         uris=song_uri,
                                                                         offset=None)  # requires list format for 'uris'

                                        except SpotifyException:
                                            speak("\nSorry, you'll need Spotify Premium to play the songs.")
                                            speak("\nIf you have Premium, make sure you've got a Spotify player open for me to utilise - such as Spotify Web or Spotify Desktop.")
                                            print("- if you have one of those open, try playing and pausing any track to test it's all working, then type the number of the song you want again.")

                                    else:
                                        speak("\nSorry, I could not find any results for tracks by the artist " + artist_name + " in the Spotify music library for you.\n")

                                except SpotifyException:
                                    speak("\nSorry, I was unable to make a search with for tracks by the artist " + artist_name + ".\n")

                        else:
                            speak("\nSorry, I could not find any results in the Spotify music library for the artist: " + search_name + ".\n")

                    except SpotifyException:
                        speak("\nSorry, I was unable to make a search with that artist information.")

    else:  # if no specific song given, choose one from global top 50 list
        try:
            global_top_50 = "https://api.spotify.com/v1/playlists/37i9dQZEVXbMDoHDwVN2tF"

            result = spotifyObject.user_playlist_tracks(user="spotifycharts", playlist_id=global_top_50, fields=None, limit=1, offset=randint(0, 49), market=None)

            if len(result['items']) != 0:

                for track in result['items']:
                    track_name = track['track']['name']
                    song_uri.append(track['track']['uri'])
                    artist_name = track['track']['artists'][0]['name']

                try:
                    speak("\nPlaying: " + track_name + " by " + artist_name)
                    spotifyObject.start_playback(device_id=None, context_uri=None, uris=song_uri,
                                                 offset=None)  # requires list format for 'uris'

                except SpotifyException:
                    speak("\nSorry, you'll need Spotify Premium to play the songs.")
                    speak("\nIf you have Premium, make sure you've got a Spotify player open for me to utilise - such as Spotify Web or Spotify Desktop.")
                    print("- if you have one of those open, try playing and pausing any track to test it's all working, then type the number of the song you want again.")

            else:
                speak("\nSorry, I could not find any results for Global Top 50 in the Spotify music library for you.\n")

        except SpotifyException:
            speak("\nSorry, I was unable to make a search for the Global Top 50.\n")

    return


def search_based_on_intent(intent_classification, named_entities_list, other_meaningful_info, user_input):
    """
    This method acts as the root method from which all search query methods for each intent are carried out.
    It also performs some filtering of data to ensure all data items are in the correct variables prior to search queries being performed.

    :param intent_classification: the intent classification of the user input text
    :type intent_classification: string
    :param named_entities_list: the list of named entities to use for the search query
    :type named_entities_list: list
    :param other_meaningful_info: the list of other meaningful info items to use for the search query
    :type other_meaningful_info: list
    :param user_input: the user input text
    :type user_input: string
    :return none
    """
    global spoken  # had to be declared explicitly as 'global' to work for some reason whereas 'song_uris' didn't
    spoken = False  # re-setting for each potential bot response

    if "r&b" in named_entities_list:  # hard-coding - but only case where '&' is used in between characters with no spaces and could be a genre (or anything other than a track)
        index_of_item = named_entities_list.index("r&b")
        if "r&b" not in other_meaningful_info:
            other_meaningful_info.append("r&b")
            del named_entities_list[index_of_item]

    if "play" in named_entities_list:  # 'Play' was being tagged as NNP and put into named_entities so removing
        index_of_item = named_entities_list.index("play")
        if "play" not in other_meaningful_info:
            other_meaningful_info.append("play")
            del named_entities_list[index_of_item]

    for item in other_meaningful_info:
        if re.search(r"\d", item):  # moves artists with number in their name (e.g. The 1975) to named entities list as were being put into other meaningful info
            index_of_item = other_meaningful_info.index(item)
            named_entities_list.append(item)
            del other_meaningful_info[index_of_item]

    for item in other_meaningful_info:  # catching artists like will.i.am which were going into other_meaningful_info
        if "." in item:
            index_of_item = other_meaningful_info.index(item)
            if item not in named_entities_list:
                named_entities_list.append(item)
                del other_meaningful_info[index_of_item]

    for item in re.findall(r"\B'(?:[^']*(?:'\b)?)+'", user_input):  # catching any terms specified explicitly in ' ' s by user, while preventing it from catching things like 's and 't in e.g. "lady's" and "don't"
        if item.lower() not in named_entities_list:
            new_item = item.strip("'")
            named_entities_list.append(new_item.lower())

    for item in named_entities_list:  # if after all of the above filtering there is a named entity in both lists, remove from other_meaningful info list
        duplicates = [x for x in other_meaningful_info if item in x]
        for dup in duplicates:
            if dup in other_meaningful_info:
                other_meaningful_info.remove(dup)

    # checks for duplicates from extraction stage e.g. 'submarine' and 'yellow submarine' - keeping longer one only
    for item in named_entities_list[:]:
        for entity in named_entities_list[:]:  # iterates over a copy of the list for safe item removal
            if item != entity and item in entity:
                named_entities_list.remove(item)

    # print("List of named entities and their types:")
    # print(entity_linking_for_named_entities(named_entities))
    # print("Other meaningful info and their types:")
    # print(entity_linking_for_other_meaningful_info(meaningful_info))

    if intent_classification == "play" and "genre" in entity_linking_for_other_meaningful_info(meaningful_info).values():  # handles case like 'Play something rock-n-roll'
        type_of_music(named_entities_list, other_meaningful_info)

    elif intent_classification == "type of music":
        type_of_music(named_entities_list, other_meaningful_info)

    elif intent_classification == "similar to":
        similar_to(named_entities_list, other_meaningful_info)

    elif intent_classification == "popularity":
        popularity(named_entities_list, other_meaningful_info)

    elif intent_classification == "artist info":
        artist_info(named_entities_list)

    elif intent_classification == "album info":
        album_info(named_entities_list)

    elif intent_classification == "track info":
        track_info(named_entities_list, other_meaningful_info)

    elif intent_classification == "play":
        play(named_entities_list)

    return


with open('XXXX', 'r') as file:  # gives access for using Google SpeechRecognition (speech-to-text)
        credential = file.read()

# Intializing some necessary global variables
spoken = False  # checking if a response has been returned by a search query method (checking if bot has "spoken")
song_uris = []
response = ""
text_mode = True
paused = False  # keeping track of if song is being played
# vocabulary associated with bot's domain - will return template message if detecting user to be talking about something not music related
keywords = ["song", "sing", "by", "style", "artist", "music", "recommend", "suggest", "playlist", "tune", "listen", "album", "track", "play",
            "songs", "sings", "styles", "singer", "artists", "recommendations", "suggestions", "playlists", "tunes", "albums", "tracks", "something"]

# The main method that is handling user input and the bot's processing of this
while True:  # While program is running

    if response == "" and not text_mode:
        r = sr.Recognizer()
        while mixer.music.get_busy():
            time.sleep(1)
        with sr.Microphone() as source:
            print("I'm listening...")
            audio = r.listen(source)
        try:
            print("I think you said: " + r.recognize_google_cloud(audio, language="en-us", credentials_json=credential))
            response = r.recognize_google_cloud(audio, language="en-us", credentials_json=credential)
        except sr.UnknownValueError:
            print("Sorry, I didn't get that. Tell me again.")
            print("Or feel free to switch to text mode by saying 'text mode'.")
            time.sleep(2)
            pass
        except sr.RequestError as e:
            print("Could not request results from Google Cloud Speech service; {0}".format(e))

    elif response == "" and text_mode:
        response = input()

    else:

        response = clean_text(response)
        # print("Cleaned text:")
        # print(response)

        if response.lower() == "done":  # makes input completely lowercase so it will work for DONE, Done, done, etc
            print("\nI'm glad I could assist you today. Bye for now.")
            break

        elif response.lower() == "show genres":
            speak("\nFor song and artist searches, these are the genres available:\n")
            song_types = spotifyObject.recommendation_genre_seeds()
            genres = []
            for genre in song_types['genres']:
                genres.append(genre)
            print(genres)

            speak("\nFor playlist searches, these are the genres available:\n")
            playlist_types = spotifyObject.categories(country=None, locale=None, limit=20, offset=0)
            categories = []
            for category in playlist_types['categories']['items']:
                categories.append(category['id'])
            print(categories)

            speak("\nFor album searches, unfortunately I haven't got a list of the genres available, so you'll just have to try out the one you're looking for.")
            response = ""

        elif bool(re.match(r'\b\d+\b', response)) and len(song_uris) != 0 and not paused:  # matches only digits in response - for song playback selection
            try:
                list_format_for_method = []  # 'uris' in start_playback method has to be a list
                list_format_for_method.append(song_uris[int(response) - 1])
                spotifyObject.start_playback(device_id=None, context_uri=None, uris=list_format_for_method, offset=None)
                print("\nFeel free to pick another number, or ask me something else.")
                print("\n-before you ask me, make sure you pause the music so I can hear you. To do so, just say or type 'pause' or 'stop'.")
                paused = False
                response = ""
                time.sleep(2)  # give it a bit of time after music starts playing

            except SpotifyException:
                speak("\nSorry, you'll need Spotify Premium to play the songs.")  # Spotify premium is required for playback method
                speak("\nIf you have Premium, make sure you've got a Spotify player open for me to utilise - such as Spotify Web or Spotify Desktop.")
                print("- if you have one of those open, try playing and pausing any track to test it's all working, then type the number of the song you want again.")
                response = ""

            except ValueError:
                speak("Sorry, that is not a valid response. Try just the number, e.g. '1'")
                response = input()

        elif response.lower() == "pause" or response.lower() == "stop" and not paused:  # for pausing playback of music
            spotifyObject.pause_playback(device_id=None)
            print("\nTo resume playback, just say or type 'play' or 'start'.")
            paused = True
            response = ""

        elif paused and response.lower() == "play" or response.lower() == "start":
            result = spotifyObject.current_playback(market=None)
            spotifyObject.seek_track(device_id=None, position_ms=result['progress_ms'])  # gets last position of playback to resume from
            spotifyObject.start_playback()
            paused = False
            response = ""

        elif response.lower() == "text mode":
            speak("Switching to text mode...")
            print("You can type it for me now. (To switch back to voice mode, just type 'voice mode'.")
            text_mode = True
            response = input()
            continue

        elif response.lower() == "voice mode":
            speak("Switching to voice mode...")
            time.sleep(1)
            text_mode = False
            response = ""

        elif response.lower() == "how are you":
            speak("\nI'm fine, thanks.")
            response = ""

        elif response.lower() == "what is your name":
            speak("\nMy name is Ava.")
            response = ""

        elif response.lower() == "hi" or response.lower() == "hello":
            speak("\nHello.")
            response = ""

        elif response.lower() == "thanks" or response.lower() == "thank you":
            speak("\nYou're welcome.")
            speak("If there's nothing else I can help you with, just say or type 'Done'.")
            response = ""

        else:

            response_words = response.split()
            found = False
            for word in response_words:
                if word.lower() in keywords:  # checks if any 'keywords' are in response
                    found = True
            if not found:
                speak("\nSorry, I cannot answer that. Maybe try re-phrasing and ask again.")

                response = ""

            else:

                print("\nHmm, let me see...\n")

                classification = classify_intent(response)
                named_entities = extract_named_entities(response)
                for entity in named_entities:
                    if "'" in entity:
                        named_entities.remove(entity)  # necessary! - for some reason it was tagging first word of strings with ' ' e.g. ['To] in 'To pimp a butterfly', so removing

                meaningful_info = extract_other_meaningful_info(response)

                for entity in meaningful_info:  # repeating above process for 'meaningful_info' list
                    if "'" in entity:
                        meaningful_info.remove(entity)

                # print("Intent classification:")
                # print(classification)

                search_based_on_intent(classification, named_entities, meaningful_info, response)

                if not spoken:
                    speak("\nSorry, I was unable to make a search based on the information you've given me. Maybe try re-phrasing and ask again.")
                    print("- Tip: You can also put things in single quotes like this: 'Swish' - this helps me to focus on it more accurately.")

                response = ""  # re-setting 'response' variable to empty to allow new processing of user input








