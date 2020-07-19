# Unit testing of main NLP methods

import ava_main_methods


def test_clean_text():
    assert ava_main_methods.clean_text("What !is a good song to listen to?") == "What  is a good song to listen to"
    assert ava_main_methods.clean_text("Give me r-n-b music") == "Give me r-n-b music"
    assert ava_main_methods.clean_text("Beyoncé songs") == "Beyonce songs"


def test_classify_intent():
    assert ava_main_methods.classify_intent("Give me some pop songs") == "type of music"
    assert ava_main_methods.classify_intent("Who sings 'Supermodel'") == "artist info"
    assert ava_main_methods.classify_intent("I want to listen to something similar to Justin Bieber") == "similar to"


def test_extract_named_entities():
    assert ava_main_methods.extract_named_entities("What is a good Michael Jackson song?") == ['michael jackson']
    assert ava_main_methods.extract_named_entities("What album is the song Thriller on?") == ['thriller']
    assert ava_main_methods.extract_named_entities("Which songs are on Will.i.am's new album?") == ['will.i.am']


def test_extract_other_meaningful_info():
    result = ava_main_methods.extract_other_meaningful_info("What are some good rock-n-roll songs?")
    assert "rock-n-roll" in result
    result = ava_main_methods.extract_other_meaningful_info("do you know any pop music?")
    assert "pop" in result


def test_entity_linking_for_named_entities():
    result = {}
    result['will.i.am'] = "artist"
    assert ava_main_methods.entity_linking_for_named_entities(['will.i.am']) == result
    result.clear()
    result['supermodel'] = "track"
    assert ava_main_methods.entity_linking_for_named_entities(['supermodel']) == result
    result.clear()
    result['damn.'] = "album"
    assert ava_main_methods.entity_linking_for_named_entities(['damn.']) == result


def test_entity_linking_for_other_meaningful_info():
    result = {}
    result['pop'] = "genre"
    assert ava_main_methods.entity_linking_for_other_meaningful_info(['pop']) == result
    result.clear()
    result['songs'] = "other_info"
    assert ava_main_methods.entity_linking_for_other_meaningful_info(['songs']) == result


