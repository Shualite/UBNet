import os 
import shutil
import time
import sys
import argparse
import json


temp_detcors = 'all_detcors.txt'
temp_res = 'evaluation_resutls.txt'

def eval_ic15(json_file="ub.json"):
	mode = 0 # 0 segm  1 kes
	os.chdir("/DATA/disk1/fsy_scenetext/ContourNet_v2/ic15_eval")
	if mode == 0:
		with open(json_file, 'r') as f:
			data = json.load(f)
			with open(temp_detcors, 'w') as f2:
					for ix in range(len(data)):
							# print('Processing: '+str(ix))
							if data[ix]['score'] > 0.1:
									outstr = '{}: {},{},{},{},{},{},{},{},{}\n'.format(data[ix]['image_id'], int(data[ix]['seg_rorect'][0]),\
															int(data[ix]['seg_rorect'][1]), int(data[ix]['seg_rorect'][2]), int(data[ix]['seg_rorect'][3]),\
															int(data[ix]['seg_rorect'][4]), int(data[ix]['seg_rorect'][5]), int(data[ix]['seg_rorect'][6]),\
															int(data[ix]['seg_rorect'][7]), \
																round(data[ix]['score'], 3))
									f2.writelines(outstr)
					f2.close()
	else:
		raise Exception('This is error.')


	########## second step: evaluate results
	dirn =  "mb_ch4_results"
	eval_dir = "./"
	# lsc = [0.51+i*1.0/100 for i in range(30)]
	lsc = [0.65]
	fres = open(temp_detcors, 'r').readlines()

	for isc in lsc:
		print ('Evaluating cf threshold {} ...'.format(str(isc)))
  
		sys.stdout.flush()
			
		if os.path.exists("mb_ch4.zip"):
			os.remove("mb_ch4.zip")
		if os.path.exists("mb_ch4_results"):
			shutil.rmtree("mb_ch4_results/")

		if not os.path.isdir(dirn):
			os.mkdir(dirn)

		for line in fres:
			line = line.strip()
			s = line.split(': ')
			filename = '{:07d}.txt'.format(int(s[0]))
			outName = os.path.join(dirn, filename)
			with open(outName, 'a') as fout:
				score = s[1].split(',')[-1].strip()
				if float(score)<isc:
					continue
				cors = ','.join(e for e in s[1].split(',')[:-1])
				fout.writelines(cors+'\n')
				
		os.chdir("mb_ch4_results/")
		os.popen("zip -r ../mb_ch4.zip ./")
		os.chdir("../")
		cmd = "python "+eval_dir+"script.py "+"-g="+eval_dir+"ch4_gt.zip -s=mb_ch4.zip"
		output = os.popen(cmd).read()
		print(output)

		if os.path.exists("mb_ch4.zip"):
			os.remove("mb_ch4.zip")
		if os.path.exists("mb_ch4_results"):
			shutil.rmtree("mb_ch4_results/")

		os.chdir("/DATA/disk1/fsy_scenetext/ContourNet_v2/")

		return output.split('\n')

if __name__ == '__main__':
	eval_ic15()