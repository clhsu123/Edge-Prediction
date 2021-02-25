from collections import defaultdict
class CountMotif:
    def count_motif(self, file_name):
        def zero():
            return 0
        file1 = open(file_name, 'r')
        lines = file1.readlines()
        #print('Number of data pieces = ' + str(len(lines)))
        count_map = defaultdict(zero)
        for l in lines:
            data = l.split('\t')
            y = data[0].split(' ')[1]
            if y == '1':
                for d in data[1:]:
                    count_map[d.split()[0]]+=int(d.split()[1])
        motif_list=sorted(count_map.keys(), key = lambda item: count_map[item], reverse = True)
        """
        print(count_map)
        print('motif list = ')
        print(motif_list)
        """
        return motif_list
