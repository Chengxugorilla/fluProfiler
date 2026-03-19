def fasta_to_dict(path):
    d = {}
    with open(path) as f:
        key = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                key = line[1:]
                d[key] = ""
            else:
                d[key] += line
    return d

def dict_to_fasta(seq_dict, fasta_path):
    """
    seq_dict: {name: seq}
    fasta_path: 输出 fasta 文件路径
    """
    with open(fasta_path, "w") as f:
        for name, seq in seq_dict.items():
            f.write(f">{name}\n")
            # 可选：按 60 列换行，方便阅读
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")
                
def make_dms(seq, start=None, end=None, aa_list="ACDEFGHIKLMNPQRSTVWY"):
    n = len(seq)
    start = 1 if start is None else start
    end = n if end is None else end

    mut_dict = {}

    for pos in range(start, end + 1):  # 1-based
        wt = seq[pos - 1]
        for aa in aa_list:
            if aa == wt:
                continue
            mut_seq = seq[:pos - 1] + aa + seq[pos:]
            key = f"{wt}{pos}{aa}"   # 例如 A205K
            mut_dict[key] = mut_seq

    return mut_dict

def make_dms_natural(ref_seq, allowed_aas, start=None, end=None):
    """
    ref_seq: 参考氨基酸序列字符串
    allowed_aas: dict, position(1-based) -> list of allowed amino acids (自然界出现过的)
    start, end: 1-based 位点区间（包含）
    """
    mutants = {}
    L = len(ref_seq)

    if start is None:
        start = 1
    if end is None:
        end = L

    # 遍历位点（1-based）
    for pos in range(start, end + 1):
        # 如果这个位点不在 allowed_aas 里，直接跳过
        if pos not in allowed_aas:
            continue

        wt = ref_seq[pos - 1]  # 序列是 0-based

        # 只在“出现过的 aa”里做突变
        for aa in allowed_aas[pos]:
            if aa == wt:
                # 不做 WT->WT 的“假突变”
                continue

            seq_list = list(ref_seq)
            seq_list[pos - 1] = aa
            mut_seq = ''.join(seq_list)

            mut_name = f"{wt}{pos}{aa}"  # 比如 A190K
            mutants[mut_name] = mut_seq

    return mutants
