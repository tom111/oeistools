# oeistools
Tools for statistics and music from OEIS.

This was described (in German) in Episode EIG046 Die Zahllücke of the [Eigenraum
podcast](https://eigenpod.de).

## Statistics

To run statistics you will need a checkout of the OEIS data, in particular the
directory `seq` from [here](https://github.com/oeis/oeisdata)

- `gap.py` iterates over the seq data and counts the occurences of each entry.
- `plot_counts.py` plots the trimmed counts from `gap.py` to produce pictures
  that look like [this classic](https://oeis.org/wiki/Frequency_of_appearance_in_the_OEIS_database).

## Music

- `sonification.py` allows to create music from a sequence. The script uses [fluidsynth](https://www.fluidsynth.org/) for which you will need a soundfont. I used `FluidR3_GM.sf2`.
