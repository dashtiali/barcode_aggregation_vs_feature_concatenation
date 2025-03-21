
from enum import Enum
import vectorization as vec

class FiltrationTypes(Enum):
    RIPS_COMPLEX = 1
    CUBICAL_COMPLEX = 2
    ALPHA_COMPLEX = 3
    LOWER_STAR = 4

class FeatureTypes(Enum):
    BETTI_CURVE = vec.GetBettiCurveFeature,
    ENTROPY_SUMMARY = vec.GetEntropySummary,
    PERS_LANDSCAPE = vec.GetPersLandscapeFeature,
    PERS_STATS = vec.GetPersStats,
    PERS_TROPICAL_COORDINATES = vec.GetPersTropicalCoordinatesFeature