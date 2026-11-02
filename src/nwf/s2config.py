# settings of study two which asks where inside the model the shape information of unnamed shapes fades and why
# every number here is fixed before any study two data and later changes are logged in STUDY2.md

# fresh seeds so the items are new to every earlier analysis and the loudness lead is tested on data it did not come from
SETS = {
    "s2_named_fill":{"kind":"named","filled":1,"named":1,"seed":201,"same_colour":False},
    "s2_named_line":{"kind":"named","filled":0,"named":1,"seed":202,"same_colour":False},
    "s2_unnamed_fill":{"kind":"blob","filled":1,"named":0,"seed":203,"same_colour":False},
    "s2_unnamed_line":{"kind":"squiggle","anchors":30,"filled":0,"named":0,"seed":204,"same_colour":False},
    # the same shapes as the two line sets but both pictures of an item share one colour pair
    "s2_named_line_same":{"kind":"named","filled":0,"named":1,"seed":205,"same_colour":True},
    "s2_unnamed_line_same":{"kind":"squiggle","anchors":30,"filled":0,"named":0,"seed":206,"same_colour":True},
}
ITEMS = 300
# the first items of every set only fit the loud directions and pick nothing else
CALIBRATION = 100
# the loud directions are fitted on the image tokens of the first items of every set
BASIS_ITEMS = 20

# the main contrast compares line drawings with and without a name so fill plays no part
MAIN_PAIR = ("s2_named_line","s2_unnamed_line")
COLOUR_PAIRS = [("s2_named_line","s2_named_line_same"),("s2_unnamed_line","s2_unnamed_line_same")]

# probe views computed at every stream index
LOUD_KS = [16,64]
RANDOM_DRAWS = 10
RANDOM_K = 64
VIEWS = ["full","ink","loud16_out","loud64_out","mean_out","ink_loud64_out"]
# a token counts as ink when at least this share of its pixels inside the shape box is ink
INK_SHARE = 0.10

# the intervention turns the loud directions down at one layer at a time on the test items of these sets
INTERVENE_SETS = ["s2_named_fill","s2_named_line","s2_unnamed_fill","s2_unnamed_line"]
INTERVENE_LAYERS = [3,9,15,21]
INTERVENE_SCALE = 0.5
INTERVENE_CONDITIONS = ["loud64","random64","uniform"]

# the models of study two and whether they get the intervention
MODELS = {"qwen":True,"qwen_names":False}

# the stream indices are split into thirds after the extra vision features are added
FIRST_STREAM = 3

# linear probe settings
PROBE_DIMS = 128
# penalty on the probe weights picked on a split of the training items
PROBE_L2 = [0.001,0.01,0.1,1.0,10.0]
