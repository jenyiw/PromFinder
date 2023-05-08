import numpy as np
import random




def read_fasta(path):
    with open(path, "r") as f:
        # Initialize variables
        seq = ""
        name = ""

        # Read the file line by line
        for line in f:
            line = line.strip()

            # If the line starts with ">", it is a name line
            if line.startswith(">"):
                if seq != "":
                    # Convert the sequence to a NumPy array
                    seq_array = np.array(list(seq))
                name = line[1:]  # Remove the ">" character
                seq = ""
            else:
                seq += line

    # Convert the last sequence to a NumPy array
    seq_array = np.array(list(seq))
    return seq_array


# print(len(read_fasta(genome_path)))

def get_background(genome_path, refTSS_path, chr_n, seed, window_size, num_positions, start_sites):
    # random.seed(seed)
    # start_sites = []

    # with open(refTSS_path, "r") as f:
    #     for line in f:
    #         print(line)
    #         if line.startswith(chr_n):  # Replace "chr1" with the chromosome you want to extract
    #             fields = line.strip().split("\t")
    #             chrom, start = fields[0], int(fields[1])
    #             start_sites.append(start)
    #             print(f"Chromosome: {chrom}, Start: {start}")
    # print(len(start_sites))

    # Avoid positions around starting sites
    avoid = set()
    for start in start_sites:
        for i in range(start - window_size, start + window_size + 1):
            avoid.add(i)
    # print(len(avoid))

    # genome_seq = read_fasta(genome_path)
    with open(genome_path, 'r') as f:
            line = f.readlines()[1:]
            item_length = len(line[0])
            total_length = item_length*(len(line)-1)+len(line[-1])
			
    positions = list(range(total_length))

    drawn_positions = []

    while len(drawn_positions) < num_positions:
        pos = random.choice(positions)
        if all(abs(start - pos) >= window_size  for start in start_sites) and pos not in avoid:
            drawn_positions.append(pos)
            for i in range(pos - window_size, pos + window_size + 1):
                avoid.add(i)
        if len(drawn_positions) % 1000 == 0:
            print(f'Selcted {len(drawn_positions)} sequences')
	
    return drawn_positions


if __name__ == "__main__":
    import os
    import pandas as pd
	
    os.chdir('..')
    genome_path = r'./human_test/genome_data/chr11.fa'

    refTSS_path = r'./human_test/cage_data/epd_human_hg38.bed'

    chr_n = "chr11"

    seed = 1
    window_size = 1000

    num_positions = 6387

    data = get_background(genome_path, refTSS_path, chr_n, seed, window_size, num_positions, 0)
	
    df = pd.DataFrame({'chr': ['chr11']*len(data),
					   'start_pos': list(data),
					   })
	
    df.to_csv('mouse_chr19_test.csv')