# -*- coding: utf-8 -*-
'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
from utils.text import cmudict

_pad = '_'
_eos = '&'
_characters = 'ABCDEFGHIJKLMçãàáâêéíóôõúûabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '
#_characters ='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '# english vocab
_punctuations = '!*(),-.:;? '#*
_phoneme_punctuations = '.!;:,?'

# Phonemes definition
'''_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
_suprasegmentals = 'ˈˌːˑ'
_other_symbols = "ʍwɥʜʢʡɕʑɺɧ'̃' "
_diacrilics = 'ɚ˞ɫ'
_phonemes = sorted(list(_vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics))

#english phonemes
_phonemes = ['l','ɹ','ɜ','ɚ','k','u','ʔ','ð','ɐ','ɾ','ɑ','ɔ','b','ɛ','t','v','n','m','ʊ','ŋ','s',
             'ʌ','o','ʃ','i','p','æ','e','a','ʒ',' ','h','ɪ','ɡ','f','r','w','ɫ','ɬ','d','x','ː',
             'ᵻ','ə','j','θ','z','ɒ']'''


#portuguese phonemes 
_phonemes =['s', 'p', 'ɡ', 'ɪ', 'z', 'f', 'ʃ', 'ɫ', 'ᵻ', 'm', 'æ', 'x', '̃', 'ː', 'r', 'ʒ', 'j', 'l', 't', 'ŋ', ';', 'i', 'ɐ', 'ɲ', 'ɑ', 'ɚ', 'w', 'd',
 'n', 'h', 'u', 'e', 'ɬ', 'ɔ', 'ɾ', 'k', 'v', 'y', 'θ', 'ʎ', 'o', 'a', 'b', 'ə', 'ɹ', 'ʊ', 'ɛ', 'ʌ']

_phonemes = sorted(list(set(_phonemes)))

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in _phonemes]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) + _arpabet
phonemes = [_pad, _eos] + list(_phonemes) + list(_punctuations)

if __name__ == '__main__':
    print(" > TTS symbols ")
    print(symbols)
    print(" > TTS phonemes ")
    print(phonemes)
