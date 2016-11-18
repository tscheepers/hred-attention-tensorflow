import os
import sys
import tarfile
import glob

AOL_ROOT_PATH = '../data/AOL-user-ct-collection'
file_ext = ".dat"


def change_dir():
    if os.getcwd().find('data') == -1:
        os.chdir(AOL_ROOT_PATH)


def get_files(f_type="bg"):
    change_dir()
    print("INFO -- searching in %s" % os.getcwd())
    files_to_process = glob.glob(f_type + "*" + file_ext)
    return files_to_process


def convert(outfile, process_one_file=None):

    if process_one_file is not None:
        change_dir()
        infile = os.getcwd() + "/" + process_one_file
        files_to_process = [infile]
    else:
        files_to_process = get_files()

    session_out = outfile + ".sess"
    session_rnk = outfile + ".rnk"
    print("INFO -- need to process %d dat-files / writing to outfile %s" % (len(files_to_process), session_out))
    with open(session_out, 'w') as sess_f,  open(session_rnk, 'w') as rnk_f:
        for filename in files_to_process:
            with open(filename, 'r') as orgf:
                print("INFO -- processing file %s" % filename)
                prevSessionID = -1
                for line in orgf:
                    SessionID, AnonID, Query = line.split('\t')[:3]
                    Query = " ".join(Query.split(","))
                    if prevSessionID != SessionID:
                        if prevSessionID == -1:
                            pass

                        else:
                            sess_f.write("\t".join(session_queries) + "\n")
                            rnk_dummy = ["1" for i in range(len(session_queries))]
                            rnk_f.write("\t".join(rnk_dummy) + "\n")
                            print(len(session_queries), len(rnk_dummy))

                        session_queries = []

                    session_queries.append(Query)
                    prevSessionID = SessionID
                # end-of-file, write out last session query
                sess_f.write("\t".join(session_queries))
                rnk_dummy = ["0" for i in range(len(session_queries))]
                rnk_f.write("\t".join(rnk_dummy) + "\n")
            if process_one_file is not None:
                print("INFO -- process only one file, exiting...")
                break
    print("INFO -- output written to %s" % outfile)

convert("session_out", process_one_file='bg_user-ct-test-collection-05.dat')
