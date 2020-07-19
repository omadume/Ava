from textblob import TextBlob  # for NLP tools / methods
from textblob.np_extractors import ConllExtractor  # for noun phrase extraction
from textblob.classifiers import NaiveBayesClassifier  # for intent classification
import csv  # for using csv files
import re  # for regular expressions
import unidecode  # for text pre-processing "cleaning" - removing diacritical marks


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
