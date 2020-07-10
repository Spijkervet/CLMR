def write_statistics(mean, std, num_songs, stats_fp):
    with open(stats_fp, "w") as f:
        f.write("mean;std;num_songs\n")
        f.write(";".join([str(mean), str(std), str(num_songs)]))