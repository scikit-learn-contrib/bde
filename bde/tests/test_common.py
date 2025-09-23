"""This file shows how to write test based on the scikit-learn common tests."""
# TODO: [@angelos, @vyron] need to pass these tests to be scikit-learn compatible
# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

from sklearn.utils.estimator_checks import parametrize_with_checks

from bde.utils.discovery import all_estimators


# parametrize_with_checks allows to get a generator of check that is more fine-grained
# than check_estimator
@parametrize_with_checks([est() for _, est in all_estimators()])
def test_estimators(estimator, check, request):
    """Check the compatibility with scikit-learn API"""
    check(estimator)
