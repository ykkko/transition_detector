DISABLE_TQDM = False

RESIZED_FRAME_HEIGHT = 360
RESIZED_FRAME_WIDTH = 640

# Frames are divided into crops. `CROP_ROWS` lines, `CROP_COLUMNS`, scale of crop = 1 / `CROP_DOWNSCALE`
CROP_ROWS = 12
CROP_COLUMNS = 12
CROP_DOWNSCALE = 12
# Note that CROP_DOWNSCALE should be more or equal then max(CROP_ROWS, CROP_COLUMNS) cause
# RESIZED_FRAME_HEIGHT must be more or equal then CROP_ROWS * (RESIZED_FRAME_HEIGHT // CROP_DOWNSCALE) and
# RESIZED_FRAME_WIDTH must be more or equal then CROP_COLUMNS * (RESIZED_FRAME_WIDTH // CROP_DOWNSCALE)


THR_SHARE_OF_CHANGED_CROPS_CUT = 0.8  # share of changed crops to say that a cut transition is detected
THR_DIFFERENCE_CROP_CUT = 0.08
THR_DIFFERENCE_FRAME_FADE = 0.05

MINIMUM_BRIGHTNESS_OF_CROP = 5  # minimum brightness of crop to remove black sequence from crops comparing

MINIMUM_LENGTH_OF_SCENE = 7  # minimum distance (in frames) between two cuts
MINIMUM_GAP_BETWEEN_FADES = 7  # minimum distance (in frames) between two fades
MINIMUM_LENGTH_OF_FADE = 5  # minimum duration (in frames) of fadein or fadeout
