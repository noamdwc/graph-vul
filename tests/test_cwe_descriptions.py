"""The checked-in inference catalog is usable without local dataset artifacts."""
from graph_vul.model.cwe_matching import load_catalog


def test_versioned_catalog_has_real_descriptions_and_excludes_unknown_markers():
    catalog = load_catalog()
    assert catalog['version'] == 'mitre-cwe-4.20-weaknesses-categories-v1'
    assert {'CWE-119', 'CWE-125', 'CWE-416', 'CWE-787'} <= catalog['descriptions'].keys()
    assert 'NONE' not in catalog['descriptions']
    assert 'NVD-CWE-Other' not in catalog['descriptions']
    assert 'NVD-CWE-noinfo' not in catalog['descriptions']


def test_catalog_scope_is_the_complete_fixed_taxonomy():
    catalog = load_catalog()
    assert catalog['source']['scope'] == {'Weaknesses': 969, 'Categories': 422}
    assert len(catalog['descriptions']) == 1391
    # Entries are included even if absent from the training/evaluation annotations.
    assert 'CWE-1447' in catalog['descriptions']
