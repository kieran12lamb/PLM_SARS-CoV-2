from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import numpy as np
import torch
import esm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from Bio import SeqIO
from scipy.special import softmax
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


## Compression Functions ########################################################
import bz2
import pickle
import _pickle as cPickle
def compressed_pickle(title, data):
  with bz2.BZ2File(title + '.pbz2', 'w') as f:
    cPickle.dump(data, f)

def decompress_pickle(file):
  data = bz2.BZ2File(file, 'rb')
  data = cPickle.load(data)
  return data
##################################################################################
## Genbank Annotation Functions ##################################################
def makeOrfTable(genbank_record):
    orfs=[]
    for feature in genbank_record.features:
        if feature.type =="CDS":
            orf = feature.qualifiers['gene'][0]
            for i, locations in enumerate(feature.location.parts):
                orfs.append([orf, locations.start, locations.end, i, locations])
    orfs = pd.DataFrame(orfs)
    orfs.columns = ['ORF','Start','End','Part','Locations']
    orfs = orfs.set_index("ORF")
    return orfs

def makeMatProteinTable(genbank_record):
    proteins=[]
    for feature in genbank_record.features:
        if feature.type =="mat_peptide":
            protein = feature.qualifiers['product'][0]
            orf = feature.qualifiers['gene'][0]
            for i, locations in enumerate(feature.location.parts):
                proteins.append([protein, orf ,locations.start, locations.end, i, locations])
    proteins = pd.DataFrame(proteins)
    proteins.columns = ['Protein',"ORF",'Start','End','Part','Locations']
    proteins = proteins.set_index("Protein")
    return proteins

###################################################################################
## Mutation Functions #############################################################
def mutate_sequence(reference_sequence,mutations):
    mutated_seq = reference_sequence
    for mutation in mutations:
        if 'ins' not in mutation and 'del' not in mutation and "X" not in mutation:
            mutant_amino = mutation[-1]
            mutant_pos = int(mutation[1:-1])
            mutated_seq = mutated_seq[:mutant_pos-1]+mutant_amino+mutated_seq[mutant_pos:]
    return mutated_seq


def DMS(reference):
  seq_list = []
  amino_acids = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
  for i,ref_amino_acid in enumerate(reference):
    for mutant_amino_acid in amino_acids:
        mutated_seq = reference[:i]+mutant_amino_acid+reference[i+1:]
        seq = SeqRecord(Seq(mutated_seq), id=ref_amino_acid+str(i+1)+mutant_amino_acid)
        seq_list.append(seq)
  return seq_list

def DMS_Table(reference):
    seq_list = []
    amino_acids = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    for i,ref_amino_acid in enumerate(reference):
        for mutant_amino_acid in amino_acids:
            mutated_seq = reference[:i]+mutant_amino_acid+reference[i+1:]
            seq = SeqRecord(Seq(mutated_seq), id=ref_amino_acid+str(i+1)+mutant_amino_acid)
            seq_list.append([ref_amino_acid+str(i+1)+mutant_amino_acid,seq])
    return pd.concat(seq_list,columns=['Mutations','Sequence'])

#####################################################################################
## Translation Functions ############################################################
def iterative_translate(sequence,truncate_proteins=False):
    amino_acid = ""
    for i in range(0,len(sequence)-2,3):
        codon = str(sequence[i:i+3])
        codon = codon.replace("?", "N")
        if "-" in codon:
            if codon == "---":
                amino_acid +="-"
            else:
                amino_acid+= "X"
        else:
            amino_acid += str(Seq(codon).translate())
    if truncate_proteins == True:
        if "*" in amino_acid:
            amino_acid = amino_acid[:amino_acid.index("*")]
    return amino_acid

def translate_with_genbank(sequence,ref):
    orfs = makeOrfTable(ref)
    translated_sequence = {orfs.index[i]+":"+str(orfs.iloc[i].Part):{"Sequence":"".join(list(iterative_translate("".join(orfs.iloc[i].Locations.extract(sequence)),truncate_proteins=True))),"ORF":orfs.index[i]} for i in range(len(orfs))}
    return translated_sequence

def translate_mat_proteins_with_genbank(sequence,ref):
    proteins = makeMatProteinTable(ref)
    proteins = proteins.drop_duplicates(subset=["ORF",'Start','End','Part',],keep="first")
    proteins_dict={}
    for i in range(len(proteins)):
        protein = "".join(list(iterative_translate("".join(proteins.iloc[i].Locations.extract(sequence)),truncate_proteins=True)))
        if proteins.index[i] in proteins_dict:
            proteins_dict[proteins.index[i]]["Sequence"] = proteins_dict[proteins.index[i]]["Sequence"]+protein
        else:
            proteins_dict[proteins.index[i]] = {"Sequence":protein, "ORF":proteins.iloc[i].ORF, "Part":proteins.iloc[i].Part}
    # translated_sequence = {proteins.index[i]:{"Sequence":"".join(list(iterative_translate("".join(proteins.iloc[i].Locations.extract(sequence)),truncate_proteins=True))), "ORF":proteins.iloc[i].ORF} }
    return proteins_dict
#####################################################################################
## ESM Embedding Functions ##########################################################
def embed_sequence(sequence,model,device,model_layers,batch_converter,alphabet):
    #Sequences to embed (We only embed the reference and use the probabilities from that to generate the scores)
    sequence_data = [('base', sequence)]

    #Get tokens etc
    batch_labels, batch_strs, batch_tokens = batch_converter(sequence_data)
    batch_len = (batch_tokens != alphabet.padding_idx).sum(1)[0]

    #Move tokens to GPU
    if torch.cuda.is_available():
        batch_tokens = batch_tokens.to(device=device, non_blocking=True)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[model_layers], return_contacts=False)
    del batch_tokens

    #Embed Sequences
    token_representation = results["representations"][model_layers][0]
    full_embedding = token_representation[1:batch_len - 1].cpu()
    base_mean_embedding  = token_representation[1 : batch_len - 1].mean(0).cpu()

    #Get Embedding and probabilities for reference sequence (Should be first sequence in data)
    lsoftmax = torch.nn.LogSoftmax(dim=1)
    base_logits = lsoftmax((results["logits"][0]).to(device="cpu"))
    return results, base_logits, base_mean_embedding,full_embedding

def process_protein_sequence(sequence,model,model_layers,batch_converter,alphabet,device):
    #Embed Sequence
    base_seq = sequence
    results,base_logits, base_mean_embedding, full_embedding = embed_sequence(base_seq,model,device,model_layers,batch_converter,alphabet)
    results_dict = {}
    results_dict["Mean_Embedding"] = base_mean_embedding.tolist()
    # results_dict["Full_Embedding"] = full_embedding.tolist()
    results_dict["Logits"] = base_logits.tolist()
    return results_dict
######################################################################################
## Scoring Functions #################################################################
def grammaticality_and_evolutionary_index(word_pos_prob, seq, mutations):
    if len(mutations) == 0:
        print('No mutations detected')
        return 0, 0
    mut_probs = []
    ev_ratios = []
    current_support = -1
    print('Mutations: ', mutations)
    for mutation in mutations:
        #Ignore insertions
        if 'ins' not in mutation and 'del' not in mutation and "X" not in mutation:
            #Split mutation 
            aa_orig = mutation[0]
            aa_pos = int(mutation[1:-1]) - 1
            aa_mut = mutation[-1]
            if (seq[aa_pos] != aa_orig):
                print(mutation)
            assert(seq[aa_pos] == aa_orig)

            #Get probabilities for changes
            prob_change = word_pos_prob[(aa_mut, aa_pos)]
            prob_original = word_pos_prob[(aa_orig, aa_pos)]
            #Log probabilities to allow for subtraction
            ev_ratio = prob_change - prob_original
            ev_ratios.append(ev_ratio)

            #Log probabilities to allow for sum rather than product
            mut_probs.append(word_pos_prob[(aa_mut, aa_pos)])
    return np.sum(mut_probs), np.sum(ev_ratios)

def semantic_calc(target,base):
    return float(sum(abs(np.array(target)-np.array(base) )))
#######################################################################################
## Genbank Functions ##################################################################
def process_sequence_genbank(sequence,genbank,model,model_layers,device,batch_converter,alphabet):
    #Translate nucleotide to proteins using genbank
    Coding_Regions= translate_with_genbank(sequence,genbank)
    Mature_Proteins= translate_mat_proteins_with_genbank(sequence,genbank)
    polyprotein_orfs =set([Mature_Proteins[prot]["ORF"] for prot in Mature_Proteins.keys()])
    Filtered_Coding_Regions = {**Coding_Regions}
    for orf in Coding_Regions.keys():
        if Coding_Regions[orf]["ORF"] in polyprotein_orfs:
            del Filtered_Coding_Regions[orf]
    Merged_Coding_Regions = {**Filtered_Coding_Regions,**Mature_Proteins}
    #Embed Sequence
    for key,value in Merged_Coding_Regions.items():
        base_seq = Merged_Coding_Regions[key]["Sequence"]
        results,base_logits, base_mean_embedding, full_embedding = embed_sequence(base_seq,model,device,model_layers,batch_converter,alphabet)
        word_pos_prob = {}
        for pos in range(len(base_seq)):
            for word in alphabet.all_toks:
                word_idx = alphabet.get_idx(word)
                prob = base_logits[pos + 1, word_idx]
                word_pos_prob[(word, pos)] = prob
        value["Mean_Embedding"] = base_mean_embedding.tolist()
        # value["Full_Embedding"] = full_embedding.tolist()
        value["Logits"] = base_logits.tolist()

    all_embeddings = [np.array(Merged_Coding_Regions [key]["Mean_Embedding"]) for key in Merged_Coding_Regions.keys()]
#     Merged_Coding_Regions ["Sum_Embedding"] = list(np.sum(all_embeddings,axis=0))
#     Merged_Coding_Regions ["Concatenated_Embedding"] = list(np.concatenate(all_embeddings))
    return Merged_Coding_Regions


def get_sequence_grammaticality(sequence,sequence_logits,alphabet):   
    prob_list = []
    sequence_logits = torch.FloatTensor(sequence_logits)
    for pos in range(len(sequence)):
        word_idx = alphabet.get_idx(sequence[pos])
        word = sequence_logits[(pos + 1,word_idx)]
        prob_list.append(word)
    base_grammaticality =np.sum(prob_list)
    return base_grammaticality


def process_and_dms_sequence_genbank(sequence,genbank,model,model_layers,device,batch_converter,alphabet,specify_orf=""):
    #Translate nucleotide to proteins using genbank
    Coding_Regions= translate_with_genbank(sequence,genbank)
    Mature_Proteins= translate_mat_proteins_with_genbank(sequence,genbank)
    polyprotein_orfs =set([Mature_Proteins[prot]["ORF"] for prot in Mature_Proteins.keys()])
    Filtered_Coding_Regions = {**Coding_Regions}
    for orf in Coding_Regions.keys():
        if Coding_Regions[orf]["ORF"] in polyprotein_orfs:
            del Filtered_Coding_Regions[orf]
    Merged_Coding_Regions = {**Filtered_Coding_Regions,**Mature_Proteins}
    embeddings = {}
    print(Merged_Coding_Regions.keys())
    if specify_orf !="":
        Merged_Coding_Regions = {specify_orf:Merged_Coding_Regions[specify_orf]}
    #Embed Sequence
    print(Merged_Coding_Regions.keys())
    for key,value in Merged_Coding_Regions.items():
        embeddings[key] = {}
        base_seq = Merged_Coding_Regions[key]["Sequence"]
        results,base_logits, base_mean_embedding, full_embedding = embed_sequence(base_seq,model,device,model_layers,batch_converter,alphabet)
        word_pos_prob = {}
        for pos in range(len(base_seq)):
            for word in alphabet.all_toks:
                word_idx = alphabet.get_idx(word)
                prob = base_logits[pos + 1, word_idx]
                word_pos_prob[(word, pos)] = prob
        embeddings[key]["Reference"] = {"Mean_Embedding":base_mean_embedding.tolist(),
                                        "Logits":base_logits.tolist(),
                                        "sequence_grammaticality":get_sequence_grammaticality(base_seq,base_logits,alphabet)
                                     }
        # Now DMS the sequence and embed and measure to reference
        sequences = DMS(base_seq)
        for fasta in sequences:
            name, sequence = fasta.id, str(fasta.seq)
            print(key,name)
            mutations = [name]
            embeddings[key][name] = process_protein_sequence(sequence,model,model_layers,batch_converter,alphabet,device)
            # L1/Manhattan Distance between mean embeddings used for the semantic change
            semantic_change = float(sum(abs(target-base) for target, base in zip(embeddings[key]["Reference"]["Mean_Embedding"],
                                                                                 embeddings[key][name] ["Mean_Embedding"])))
            gm, ev = grammaticality_and_evolutionary_index(word_pos_prob, base_seq, mutations)
#             print('Semantic score: ', semantic_change)
#             print('Grammaticality: ', gm)
#             print('Relative Grammaticality: ', ev)
            embeddings[key][name]["label"] = name
            embeddings[key][name]["semantic_score"] = semantic_change
            #Probability of mutation, given the reference sequence
            embeddings[key][name]["grammaticality"] = gm
            embeddings[key][name]["relative_grammaticality"] = ev
            #Probability of whole sequence
            embeddings[key][name]['sequence_grammaticality'] = get_sequence_grammaticality(sequence,embeddings[key][name]['Logits'],alphabet)
#             print('Sequence Grammaticality: ', embeddings[key][name]['sequence_grammaticality'])
            #Probability ratio between the mutant sequence and the reference sequence
            embeddings[key][name]['relative_sequence_grammaticality'] = embeddings[key][name]['sequence_grammaticality']-embeddings[key]["Reference"]['sequence_grammaticality']
#             print('Relative Sequence Grammaticality: ', embeddings[key][name]['relative_sequence_grammaticality'])
            embeddings[key][name]["probability"] = np.exp(gm)
#             print(embeddings[key][name]['grammaticality'])
    return embeddings


def get_mutations(seq1, seq2):
    mutations = []
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            if seq1[i] != '-' and seq2[i] == '-':
                mutations.append('{}{}del'.format(seq1[i], i + 1))
            else:
                mutations.append('{}{}{}'.format(seq1[i] , i + 1, seq2[i]))
    return mutations

def get_indel_mutations(aligned_reference,indel_sequence):
    index = 1
    insertion_subtract = 0
    mutations =[] 
    insertions = []
    deletions = []
    for i,a in enumerate(aligned_reference):
        position = index+i - insertion_subtract
        if a == '-' and indel_sequence[i]!= '-' :
            insertions.append('ins'+str(position)+indel_sequence[i])
            insertion_subtract +=1
        elif a == '-' and indel_sequence[i]== '-' :
            insertion_subtract +=1
        elif  a != '-' and indel_sequence[i] == '-':
            deletions.append('del'+str(position))
        elif aligned_reference[i] != indel_sequence[i]:
            mutations.append(aligned_reference[i]+str(position)+indel_sequence[i])
    return mutations,insertions,deletions

#######################################################################################
## Fasta Functions ####################################################################
def process_fasta(filename,protein_name,reference_sequence,model,model_layers,batch_converter,device,alphabet,insertions=False):
    key = protein_name
    base_seq = reference_sequence  
    embeddings = {key:{}}
    results,base_logits, base_mean_embedding, full_embedding = embed_sequence(base_seq,model,device,model_layers,batch_converter,alphabet)
    # Get position probabilities dictionary
    word_pos_prob = {}
    for pos in range(len(base_seq)):
        for word in alphabet.all_toks:
            word_idx = alphabet.get_idx(word)
            prob = base_logits[pos + 1, word_idx]
            word_pos_prob[(word, pos)] = prob
            
    embeddings[key]["Reference"] = {"Mean_Embedding":base_mean_embedding.tolist(),
                                    "Logits":base_logits.tolist(),
                                    "sequence_grammaticality":get_sequence_grammaticality(base_seq,base_logits,alphabet)}
    for fasta in SeqIO.parse(filename, "fasta"):
        name, sequence = fasta.id, str(fasta.seq)
        if insertions == False:
            mutations = get_mutations(reference_sequence,str(fasta.seq))
        if sequence[-1] == '*':
            sequence = sequence[:-1]
        
        #Remove gap characters
        sequence = sequence.replace('-','')
        
        embeddings[key][name] = process_protein_sequence(sequence,model,model_layers,batch_converter,alphabet,device)
        # L1/Manhattan Distance between mean embeddings used for the semantic change
        semantic_change = float(sum(abs(target-base) for target, base in zip(embeddings[key]["Reference"]["Mean_Embedding"],
                                                                             embeddings[key][name] ["Mean_Embedding"])))
        
        if insertions == False:
            gm, ev = grammaticality_and_evolutionary_index(word_pos_prob, base_seq, mutations)
            #Probability of mutation, given the reference sequence
            embeddings[key][name]["grammaticality"] = gm
            embeddings[key][name]["relative_grammaticality"] = ev
            embeddings[key][name]["probability"] = np.exp(gm)
            print('Grammaticality: ', gm)
            print('Relative Grammaticality: ', ev)
        
        embeddings[key][name]["label"] = name
        embeddings[key][name]["semantic_score"] = semantic_change
        print('Semantic score: ', semantic_change)

        #Probability of whole sequence
        embeddings[key][name]['sequence_grammaticality'] = get_sequence_grammaticality(sequence,embeddings[key][name]['Logits'],alphabet)
        print('Sequence Grammaticality: ', embeddings[key][name]['sequence_grammaticality'])
        #Probability ratio between the mutant sequence and the reference sequence
        embeddings[key][name]['relative_sequence_grammaticality'] = embeddings[key][name]['sequence_grammaticality']-embeddings[key]["Reference"]['sequence_grammaticality']
        print('Relative Sequence Grammaticality: ', embeddings[key][name]['relative_sequence_grammaticality'])
        
    return embeddings
#######################################################################################
## VOC Functions ####################################################################
def build_voc_dictionary(lineage_dict):
    vocs = {"Alpha":["B.1.1.7"],"Beta":["B.1.351"],"Gamma":["P.1"],"Delta":["B.1.617.2"],"Omicron":["B.1.1.529"]}
    
    for key in lineage_dict.keys():
        val = lineage_dict[key]
        if type(val) == str:
            val = [val]
            
        for lineage in val:
            if "." in lineage:
                start  = lineage.split(".")[0]
            else:
                start  = lineage
            if "B.1.617.2" == lineage or "B.1.617.2." in lineage or start in vocs["Delta"] :
                vocs["Delta"].append(key)
            elif "B.1.1.7" == lineage or  "B.1.1.7." in lineage or start in vocs["Alpha"]:
                vocs["Alpha"].append(key)
            elif "B.1.351" == lineage or "B.1.351." in lineage or start in vocs["Beta"]:
                vocs["Beta"].append(key)
            elif "P.1" == lineage or  lineage == "P.1" or "P.1." in lineage or start in vocs["Gamma"]:
                vocs["Gamma"].append(key)
            elif "B.1.1.529" == lineage or "B.1.1.529." in lineage or start in vocs["Omicron"]:
                vocs["Omicron"].append(key)
    return vocs

def is_voc(lineage,voc_dictionary):
    for key,values in voc_dictionary.items():
        if "." in lineage:
            start  = lineage.split(".")[0]
        else:
            start  = lineage
        #If base VOC lineage 
        if lineage in values:
            return key
        #If not the base VOC
        else:
            if start in values:
                return key
            elif values[0]+"." in lineage:
                return key
    return "Non-VOC"


