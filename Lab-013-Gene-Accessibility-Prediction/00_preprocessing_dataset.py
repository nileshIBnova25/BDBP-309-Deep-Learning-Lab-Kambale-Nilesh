"""
Lab 13: Gene Accessibility Prediction Using CNNs

Data: HCT116 DNase-seq peaks from FILER and the hg38 chromosome 1 reference genome from UCSC.
Dataset: 6,574 DNA sequences (600 bp each), labelled as peak-associated (1) or no detected peak (0).
Objective: Train a CNN to classify DNA sequences based on DNase-seq peak association.

FILER: https://filer2.niagads.org/Annotationtracks/ENCODE/data/DNase-seq/narrowpeak/hg38/ENCFF676ZZP.bed.gz
UCSC: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz
"""
#----------------------Imports----------------------#
import gzip
import random
import csv
#-----------------------------------------------------#

# 1) Load DNA Sequence

genome_file = "/home/ibab/DL-Lab/Data/Lab13_Gene_Accessibility/chr1.fa.gz"
with gzip.open(genome_file,"rt") as f:
    genome = "".join(line.strip() for line in f if not line.startswith(">"))
print(genome)
print("Genome length", len(genome))

#------------------------------------------------------#
# 2) Load DNase peaks

peak_file = "/home/ibab/DL-Lab/Data/Lab13_Gene_Accessibility/HCT116_DNase_peaks.bed.gz"
peaks = []

with gzip.open(peak_file, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        row = line.strip().split("\t")

        if row[0] == "chr1":
            start = int(row[1])
            end = int(row[2])
            summit = int(row[9])

            if summit == -1:
                summit = (start + end) // 2
            else:
                center = start + summit

            peaks.append((center, start, end))
print("Total peaks:", len(peaks))
#----------------------------------------------------------------#
# 3) Extract positive sequences

random.seed(42)
sequences = []

for center, peak_start, peak_end in peaks:
    start = center - 300
    end = center + 300

    seq = genome[start:end]

    if len(seq) == 600 and set(seq) <= set("ACGT"):
        sequences.append([seq, 1, start, end])
random.shuffle(sequences)
sequences = sequences[:5000]
print("Positive sequences", len(sequences))

#-------------------------------------------------#
# 4) Extract Negatives sequences

negative = []

while len(negative) < len(sequences):
    center = random.randint(300, len(genome) - 300)
    start = center - 300
    end = center + 300

    # Avoid regions within 1000 bp of any detected peak
    if any(start < p[2] + 1000 and end > p[1] - 1000 for p in peaks):
        continue

    # Avoid overlapping negative sequences
    if any(abs(center - x[3]) < 600 for x in negative):
        continue
    seq = genome[start:end]

    if len(seq) == 600 and set(seq) <= set("ACGT"):
        negative.append([seq, 0, start, end])

    if len(negative) >= 5000:
        break
print("Negative sequences", len(negative))

#-----------------------------------------------------------------------#

# 5) Save dataset

data = sequences + negative
random.shuffle(data)
with open("/home/ibab/DL-Lab/Data/Lab13_Gene_Accessibility/binary_accessibility_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sequence", "label", "start", "end"])
    writer.writerows(data)
print("Dataset saved successfully!")
print("Total sequences:", len(data))

#----------------------------------------------------------------------#


