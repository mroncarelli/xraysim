import os
import warnings

import numpy as np
import pytest
from astropy.io import fits

# List of environment variables that may appear in the path of files written in FITS headers
environmentVariablesList = ['XRAYSIM', 'SIXTE']
environmentVariablesPathList = [os.environ.get(envVar) for envVar in environmentVariablesList]


def history_unpack(history: list) -> list:
    """
    Modifies a HISTORY list record by appending lines that correspond to the same record.
    :param history: (list of str) HISTORY record.
    :return: (list of str) Modified HISTORY record.
    """

    result = []
    for index, record in enumerate(history):
        if index > 0 and record.startswith('P') and record.split(' ')[0] == history[index - 1].split(' ')[0]:
            result[-1] += record[record.index(' ') + 1:]
        else:
            result.append(record)
    return result


def assert_string_in_header_matches_reference(string: str, string_reference: str) -> None:
    """
    Checks that a string in a header matches a reference string. If the string contains an environment variable it may
    differ from the reference one, but the test should not fail: in this case the test checks that the parts before and
    after the environment variable match. Otherwise, it checks that the two sting are identical.
    :param string: (str) String to check.
    :param string_reference: (str) Reference string.
    :return: None.
    """

    for envVar in environmentVariablesPathList:
        if envVar in string:
            split_list = string.split(envVar)
            assert string_reference.startswith(split_list[0]) and string_reference.endswith(split_list[-1])
            return None
        else:
            pass
    assert string == string_reference
    return None


def assert_header_has_all_keywords_and_values_of_reference(header: fits.header, header_reference: fits.header,
                                                           key_skip=None, history_tag_skip=None) -> None:
    """
    Checks that a header contains all the keys, with same values, than the reference one. Other keywords/values may
    be present and do not affect the result.
    :param header: (fits.header) Header to check.
    :param header_reference: (fits.header) Reference header.
    :param key_skip: (tuple or list) List of header keywords should not be checked.
    :param history_tag_skip: (tuple or list) List of tags that if present in a history record should not be checked.
    :return: None.
    """

    history_checked = False
    for key in header_reference.keys():
        val_reference = header_reference.get(key)
        assert key in header
        val = header.get(key)
        if key_skip is not None and key in key_skip:
            pass
        elif key == 'HISTORY':
            # The 'key' iteration goes through all history records one by one, but 'val' always contains the
            # whole history, so only one check is needed
            if not history_checked:
                history = history_unpack(val)
                history_reference = history_unpack(val_reference)

                for history_record, history_record_reference in zip(history, history_reference):
                    if any(tag in history_record_reference for tag in history_tag_skip):
                        pass
                    else:
                        assert_string_in_header_matches_reference(history_record, history_record_reference)
                history_checked = True
        else:
            if type(val) == str:
                assert_string_in_header_matches_reference(val, val_reference)
            else:
                assert val == pytest.approx(val_reference)
    return None


def assert_data_matches_reference(inp, reference) -> None:
    """
    Checks that data in an HDU match reference.
    :param inp: Input data.
    :param reference: Reference data.
    :return: None.
    """

    # Checking that the two objects are of the same type
    assert isinstance(inp, type(reference))

    if isinstance(inp, type(None)):
        # Nothing to do here since the type check already done ensures also inp is None
        pass

    elif isinstance(inp, np.ndarray):
        if isinstance(inp, fits.fitsrec.FITS_rec):
            # Checking number of rows and field names
            assert len(inp) == len(reference)
            assert inp.dtype.names == reference.dtype.names
            # Checking data iterating by field and vale
            for name in inp.dtype.names:
                for val, val_reference in zip(inp[name], reference[name]):
                    assert val == pytest.approx(val_reference)
        else:
            # Checking all values
            assert np.all(inp == pytest.approx(reference))

    else:
        warnings.warn("Found unknown data type in FITS hdu:" + str(type(inp)))

    return None


def assert_hdu_list_matches_reference(inp: fits.hdu.hdulist.HDUList, reference: fits.hdu.hdulist.HDUList, key_skip=None,
                                      history_tag_skip=None) -> None:
    """
    Checks that a FITS file matches a reference one
    :param inp: (HDUList) HDUList to check.
    :param reference: (HDUList) Reference HDUList.
    :param key_skip: (tuple or list) List of header keywords should not be checked.
    :param history_tag_skip: (tuple or list) List of tags that if present in a history record should not be checked.
    :return: None.
    """

    for hdu, hdu_reference in zip(inp, reference):
        assert_header_has_all_keywords_and_values_of_reference(hdu.header, hdu_reference.header, key_skip=key_skip,
                                                               history_tag_skip=history_tag_skip)
        assert_data_matches_reference(hdu.data, hdu_reference.data)

    return None
