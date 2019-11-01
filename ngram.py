# -*- coding: UTF-8 -*-

import os, re, glob, unicodedata, time
from collections import Counter
from pathlib import PurePath
import h5py
import numpy as np

verbose = False
#verbose = 1

langs = {
'af':'Afrikaans',
'als':'Alsatian',
'ar':'Arabic',
'az':'Azerbaijan',
'be':'Belorussian',
'bg':'Bulgarian',
'bi':'Bislama (currently also used by Bitruscan and Tok Pisin)',
'bn':'Bengali',
'br':'Breton',
'bs':'Bosnian',
'ca':'Catalan',
'cdo':'Cantonese (Roman script)',
'chr':'Cherokee',
'co':'Corsican',
'cs':'Czech',
'csb':'Kashubian',
'cy':'Welsh',
'da':'Danish',
'de':'German',
'dk':'Danish',
'dv':'Dhivehi',
'el':'Greek',
'en':'English',
'eo':'Esperanto',
'es':'Spanish',
'et':'Estonian',
'eu':'Basque',
'fa':'Persian',
'fi':'Finnish',
'fo':'Faroese',
'fr':'French',
'fy':'Frisian',
'ga':'Irish Gaelic',
'gd':'Scottish Gaelic',
'gl':'Galician',
'gn':'Guarani',
'gu':'Gujarati',
'gv':'Manx',
'he':'Hebrew',
'hi':'Hindi',
'hr':'Croatian',
'hu':'Hungarian',
'ia':'Interlingua',
'id':'Indonesian',
'io':'Ido',
'is':'Icelandic',
'it':'Italian',
'ja':'Japanese',
'jv':'Javanese',
'ka':'Georgian',
'km':'Khmer',
'ko':'Korean',
'ks':'Ekspreso, but should become Kashmiri',
'ku':'Kurdish',
'la':'Latin',
'lt':'Latvian',
'lv':'Livonian',
'mg':'Malagasy',
'mi':'Maori',
'minnan':'Min-Nan',
'mk':'Macedonian',
'ml':'Malayalam',
'mn':'Mongolian',
'mr':'Marathi',
'ms':'Malay',
'na':'Nauruan',
'nah':'Nahuatl',
'nb':'Norwegian (Bokmal)',
'nds':'Lower Saxon',
'nl':'Dutch',
'no':'Norwegian',
'nn':'Norwegian (Nynorsk)',
'oc':'Occitan',
'om':'Oromo',
'pl':'Polish',
'ps':'Pashto',
'pt':'Portuguese',
'ro':'Romanian',
'roa-rup':'Aromanian',
'ru':'Russian',
'sa':'Sanskrit',
'sh':'Serbocroatian',
'si':'Sinhalese',
'simple':'Simple English',
'sk':'Slovakian',
'sl':'Slovenian',
'sq':'Albanian',
'sr':'Serbian',
'st':'Sesotho',
'su':'Sundanese',
'sv':'Swedish',
'sw':'Swahili',
'ta':'Tamil',
'th':'Thai',
'tl':'Tagalog',
'tlh':'Klingon',
'tokipona':'Toki Pona',
'tpi':'Tok Pisin',
'tr':'Turkish',
'tt':'Tatar',
'uk':'Ukrainian',
'ur':'Urdu',
'uz':'Uzbek',
'vi':'Vietnamese',
'vo':'Volapuk',
'wa':'Walon',
'xh':'isiXhosa',
'yi':'Yiddish',
'yo':'Yoruba',
'wo':'Wolof',
'za':'Zhuang',
'zh':'Chinese',
'zh-cn':'Simplified Chinese',
'zh-tw':'Traditional Chinese',
'pcm':'Naija',
'arz':'Arabizi'
}

typical={}
typical['en']=("anoda comot dem dey di dia don doti everitin im na pickin pickins pikin pesin waka wetin wen wan wella wey".strip().split(),'pcm')
typical['fr']=("3ada 3adi 3adim 3adna 3al 3ala 3alam 3alamine 3alayhi 3alaykom 3alaykoum 3alem 3ali 3alik 3alikom 3alikoum 3alina 3am 3ame 3ami 3an 3ana 3and 3andak 3ande 3andek 3andha 3andhom 3andhoum 3andi 3andkom 3andna 3ando 3andou 3ans 3antar 3arab 3ayb 3aybe 3aychin 3aychine 3ib 3ibad 3la 3lach 3lah 3lih 3liha 3lihom 3lihoum 3lik 3likom 3likoum 3lina 7na a3la ba3d cha3b cha3be echa3b el3am ga3 jami3 jami3a jma3a l3ada l3am la3ab m3a m3ah m3aha m3ahom m3ahoum m3ak m3ake m3akom m3akoum m3ana ma3a ma3ak m3ana ma3lich me3a ni3ma nta3 sa3dan sa3dane ta3 ta3e ta3lik ta3na tal3a tal3ab te3 tel3ab ya3arfou ya3tik yal3ab yel3ab za3ma ana ghir hadi hata hna khir rak rana wach wala walah wallah".strip().split(),'arz')
typical['mg']=("3ada 3adi 3adim 3adna 3al 3ala 3alam 3alamine 3alayhi 3alaykom 3alaykoum 3alem 3ali 3alik 3alikom 3alikoum 3alina 3am 3ame 3ami 3an 3ana 3and 3andak 3ande 3andek 3andha 3andhom 3andhoum 3andi 3andkom 3andna 3ando 3andou 3ans 3antar 3arab 3ayb 3aybe 3aychin 3aychine 3ib 3ibad 3la 3lach 3lah 3lih 3liha 3lihom 3lihoum 3lik 3likom 3likoum 3lina 7na a3la ba3d cha3b cha3be echa3b el3am ga3 jami3 jami3a jma3a l3ada l3am la3ab m3a m3ah m3aha m3ahom m3ahoum m3ak m3ake m3akom m3akoum m3ana ma3a ma3ak m3ana ma3lich me3a ni3ma nta3 sa3dan sa3dane ta3 ta3e ta3lik ta3na tal3a tal3ab te3 tel3ab ya3arfou ya3tik yal3ab yel3ab za3ma ana ghir hadi hata hna khir rak rana wach wala walah wallah".strip().split(),'arz')


#rtlLanguages=['Arabic','Hebrew','Pashto','Persian','Urdu']
#decPunctex=re.compile(r"\d*\W*", re.U+re.I)



class Ngram:

	def __init__(self, make_new = False, ngramnum = 500):
		self.n=3
		self.ngramnum = ngramnum
		self.charnum = 20
		self.matrix_ngram = None

		self.freqmulti = 1000000

		self.langfolder = "./data/languages"
		self.langext = "lang.txt"
		self.ngramext = "ng"
		self.ngrams={}
		self.replare = re.compile(r"\s+")
		self.reapostrophe = re.compile(r"’")
		self.renotword = re.compile(r"[\W\d_]+")
		#self.repoint=re.compile(r'(?<![0-9A-ZÀÈÌÒÙÁÉÍÓÚÝÂÊÎÔÛÄËÏÖÜÃÑÕÆÅÐÇØ])([.。\!\?\n\r]+)')
		self.repoint=re.compile(r'[.。!?]*[\n\r]+|(?<![0-9A-ZÀÈÌÒÙÁÉÍÓÚÝÂÊÎÔÛÄËÏÖÜÃÑÕÆÅÐÇØ])([.。\!\?]+)')
		#self.repoint=re.compile(r'(?<![0-9A-ZÀÈÌÒÙÁÉÍÓÚÝÂÊÎÔÛÄËÏÖÜÃÑÕÆÅÐÇØ])([.。\!\?\n\r]+)')


		self.charGuesser = 	{
		("Lo","CJK"):"zh",
		("Lo","HANGUL"):"ko",
		("Lo","HIRAGANA"):"ja",
		("Lo","KATAKANA"):"ja",
		("Ll","GREEK"):"el",
		("Lu","GREEK"):"el",
		("Lo","GUJARATI"):"gu",
		("Lo","GEORGIAN"):"ka",
		("Lo","BENGALI"):"bn",
		("Lo","TAMIL"):"ta",
		("Lo","THAI"):"th",
		("Lo","THAANA"):"dv",
		("Lo","DEVANAGARI"):"ngram",
		("Ll","CYRILLIC"):"ngram",
		("Lu","CYRILLIC"):"ngram",
		("Lo","ARABIC"):"ngram",
		("Lo","ARABIC"):"ngram",
		("Ll","LATIN"):"ngram",
		("Lu","LATIN"):"ngram"
		}

		if make_new:
			print('Making new ngram in {}'.format(self.langfolder))
			self.makeNgrams()
			self.NgramsToMatrix()

		else:
			with h5py.File('data/matrix_ngram_data.h5', 'r') as hf:
				matrix_ngram = hf['dataset_1'][:]
				self.matrix_ngram = matrix_ngram/matrix_ngram.sum(axis=1, keepdims=True)

			self.list_sorted_languages = []
			with open('data/list_language_sorted.txt', 'r+', encoding='utf-8') as f_lang:
				for lang in f_lang:
					self.list_sorted_languages.append(lang.rstrip('\n'))

			self.list_sorted_ngram = []
			with open('data/list_sorted_ngram.txt', 'r+', encoding='utf-8') as f_ngram:
				for ngram in f_ngram:
					self.list_sorted_ngram.append(ngram.rstrip('\n'))
					self.set_ngram = set(self.list_sorted_ngram)
					self.dict_ngram_index = dict()
					for n, ngram in enumerate(self.list_sorted_ngram):
						self.dict_ngram_index[ngram] = n



		# self.readNgrams()
		# self.NgramsToMatrix()
		# if import_matrix == True:
		# 	# --> add assert if file not in folder <--




	#def guessLanguageRealName(self,text):
		#"""
		#wrapper class for guessLanguage
		#gives back real name
		#"""

		#return langs.get(self.guessLanguage(text)[1],"an unknown language")#, maybe "+str(self.mostCommonUnicodeKeys(text)))

	def makeNgrams(self):

		folder = os.path.join(self.langfolder,'*'+self.langext)
		print("making",str(self.n)+"-grams...")
		number=0
		for infilename in glob.glob(os.path.normcase(folder)): # for each minicorpus of a language
			# language = infilename.split("/")[-1].split(".")[0]
			language = PurePath(infilename).parts[-1].split(".")[0]
			infile = open(infilename, encoding='utf-8')
			outfile = open(os.path.join(self.langfolder,language+"."+self.ngramext) ,"w" , encoding='utf-8')
			text = infile.read()
			textlen=len(text)
			text = self.reapostrophe.sub("'",self.replare.sub("_",(" "+text+" ").lower()))

			if textlen==0:continue
			#invlen = 1.0/textlen

			#print text

			sordico = self.ngramList(text,self.n)[:self.ngramnum]
			# number_ngram = len(text) - self.n + 1
			#sordico = sordico
			#self.ngrams[language]=sordico
			#print len(text)

			for f,g in sordico:
				outfile.write(g+"\t"+str(f)+"\n")

			# print('write', infilename)
			number+=1

		print("  done",number,"languages")

	def ngramList(self,text,n):
		"""
		for a text, the size n of the ngram and the value to add for each ngram found,
		the function gives back
		a list of couples
		"""
		thesengrams={}
		for i in range(len(text)-n):
			nuple = text[i:i+n]
			thesengrams[nuple]=thesengrams.get(nuple,0)+1
		sordico = [(f,g) for g,f in thesengrams.items()]
		sordico.sort()
		sordico.reverse()
		#print("sordico",sordico)
		return sordico


	def readNgrams(self):
		folder = os.path.join(self.langfolder,'*'+self.ngramext)
		for filename in glob.glob(os.path.normcase(folder)):
			# language = filename.split("/")[-1].split(".")[0]
			# language = filename.split("\\")[-1].split(".")[0]
			language = PurePath(filename).parts[-1].split(".")[0]

			self.ngrams[language]={}
			file = open(filename, encoding='utf-8')
			for line in file:
				try:
					g,f=line.split("\t")
					self.ngrams[language][g]=int(f)
				except:
					print("error in file",filename)

	def NgramsToMatrix(self):
		if not self.ngrams:
			self.readNgrams()



		list_all_ngram = []
		for l in self.ngrams:
			list_all_ngram.extend(list(self.ngrams[l].keys()))

		set_ngram = set(list_all_ngram)
		self.list_sorted_ngram = sorted(list(set_ngram))
		self.set_ngram = set(self.list_sorted_ngram)
		self.dict_ngram_index = dict()
		for n, ngram in enumerate(self.list_sorted_ngram):
			self.dict_ngram_index[ngram] = n

		# list_unsorted_ngram = list(set_ngram)

		self.list_sorted_languages = sorted(list(self.ngrams.keys()))
		self.set_languages = set(self.list_sorted_languages)
		# list_unsorted_languages = list(self.ngrams.keys())

		matrix_ngram = np.zeros((len(self.list_sorted_languages),len(self.list_sorted_ngram)))

		for language in self.ngrams.keys():
			for ngram, count in self.ngrams[language].items():
				matrix_ngram[binary_search(self.list_sorted_languages, language),binary_search(self.list_sorted_ngram, ngram)] = count

		self.matrix_ngram = matrix_ngram/matrix_ngram.sum(axis = 1,keepdims=True)

		# write the matrix on local disk
		with h5py.File('data/matrix_ngram_data.h5', 'w') as h5f:
			h5f.create_dataset('dataset_1', data=matrix_ngram)

		with open('data/list_language_sorted.txt', 'w+', encoding='utf-8') as f_lang:
			for lang in self.list_sorted_languages:
				f_lang.write(lang+'\n')

		with open('data/list_sorted_ngram.txt', 'w+', encoding='utf-8') as f_ngram:
			for ngram in self.list_sorted_ngram:
				f_ngram.write(ngram+'\n')


	def sample_predict(self, sampletext):
		"""
		central function
		predict the language of a sample text
		returns unknown if no characters that allow guessing

		"""
		uniresult = self.char_guesser(sampletext)
		if  uniresult != "ngram":
			return uniresult

		sampletext = self.replare.sub("_",(" "+sampletext+" ").lower())
		samplengrams = Counter([sampletext[i:i+self.n] for i in range(len(sampletext)-self.n+1)])
		resultcode = self.dot_prod(samplengrams)

		if resultcode in typical:
			resultcode = self.checktypical(sampletext, resultcode, typical[resultcode])
		return resultcode

	def chunck_predict(self, chunk, threshold = 0.01):
		"""Return two list *list_1* and *list_2* :
		- list_1 : predictions for each line of the chunk
		- list_2 : score of the prediction for each of these lines"""

		number_of_line = len(chunk)
		languages_list = ['unknown']*number_of_line
		score_list = [0]*number_of_line
		vectors_list = []
		list_index = []
		# number_of_char = 0
		for n, line in enumerate(chunk):
			# number_of_char += len(line)
			#         if len(line) < 10:continue
			uniresult = self.char_guesser(line)
			languages_list[n] = uniresult
			if uniresult != 'ngram':
				continue
			list_index.append(n)
			samplevector = self.text_to_vector(line, uniresult)
			vectors_list.append(samplevector)


		concat_vec = np.concatenate(vectors_list)
		dot_prod = np.dot(self.matrix_ngram,concat_vec.T)
		dot_prod_argmax = np.argmax(dot_prod, axis = 0)
		for n, index in enumerate(list_index):
			local_argmax = dot_prod_argmax[n]
			if dot_prod[local_argmax, n]<=threshold:
				resultcode = 'unknown'
			else:
				resultcode = self.list_sorted_languages[local_argmax]
				score_list[index] = dot_prod[local_argmax, n]

				if resultcode in typical:
					resultcode = self.checktypical(line, resultcode, typical[resultcode])

			languages_list[index] = resultcode
		return languages_list, score_list


	def char_guesser(self, sampletext):
		cs = Counter(sampletext).most_common(self.charnum)
		unicat,_ = Counter([unicodedata.category(char) for char,count in cs]).most_common(1)[0]
		uniname,_ = Counter([unicodedata.name(c,"_").split()[0] for c,i in cs]).most_common(1)[0]

		#print("////mostCommonUnicodeKeys:",unicat,uniname)
		uniresult = self.charGuesser.get((unicat,uniname),"unknown")

		return uniresult


	def text_to_vector(self, sampletext, uniresult = 'ngram'):
		if uniresult != 'ngram':
			return np.zeros((1, len(self.list_sorted_ngram)))

		sampletext = self.replare.sub("_",(" "+sampletext+" ").lower())
		samplengrams = Counter([sampletext[i:i+self.n] for i in range(len(sampletext)-self.n+1)])
		samplevector = np.zeros((1, len(self.list_sorted_ngram)))
		for ngram in samplengrams.keys():
			if ngram not in self.set_ngram:
				continue
			else:
				ngram_index = self.dict_ngram_index[ngram]
				# ngram_index = binary_search(self.list_sorted_ngram, ngram)
				samplevector[0, ngram_index] = samplengrams[ngram]

		return samplevector



	def simpledistance(self, samplengrams, addValue):

		for ng in samplengrams.keys():
			samplengrams[ng] = samplengrams[ng] * addValue
		distdico={}
		for l,ng in self.ngrams.items(): # for each language
			for nuple, f in samplengrams.items():#.iteritems(): # for each ngram
				distdico[l]= distdico.get(l,0) + abs(ng.get(nuple,0)-f)
		return distdico

	def dot_prod(self, samplengrams):
		# --> Add make or read matrix function <--
		sample_ngram_vector = np.zeros((1, len(self.list_sorted_ngram)))
		for ngram in samplengrams.keys():
			ngram_index = binary_search(self.list_sorted_ngram, ngram)
			if  ngram_index != -1 :
			#             N += dict_trigram[trigram]
				sample_ngram_vector[0, ngram_index] = samplengrams[ngram]

		if sample_ngram_vector.sum() == 0: return "unknown"

		norm_vec = (sample_ngram_vector/sample_ngram_vector.sum(axis = 1, keepdims=True))
		# norm_vec = sample_ngram_vector
		dot_prod = np.dot(self.matrix_ngram,norm_vec.T)
		argmax = np.where(dot_prod == np.amax(dot_prod))
		resultcode = self.list_sorted_languages[argmax[0][0]]
		# for n ,lang in enumerate(self.list_sorted_languages):
		# 	print(lang,dot_prod[n])
		# print(samplengrams)

		return resultcode


	def checktypical(self, sampletext, resultcode, wordscode):
		"""
		function looking for typical words of a language

		"""
		sampletext=sampletext[1:-1]
		#print('checktypical',sampletext)
		words, newcode = wordscode
		#print(set(self.renotword.split(sampletext)) , set(words),set(self.renotword.split(sampletext)) & set(words))
		if len(set(self.renotword.split(sampletext)) & set(words)): # non empty intersection: found a typical word
			#print(89798798,newcode)
			return newcode
		return resultcode





	#def allLanguages(self):
	"""
	pour gromoteur

	"""
		#"""
		#gives a list of all languages we have information about
		#"""
		#codes = []
		#names = []
		#for cat,key,name in self.charGuesser:
			#if name != "ngram":codes+=[name]


		#folder = os.path.join(self.langfolder,'*'+self.ngramext)
		#for filename in glob.glob(os.path.normcase(folder)):
			#language = filename.split("/")[-1].split(".")[0]
			#codes+=[language]

		#codes=list(set(codes))
		#codes.sort()
		#for c in codes:
			#names+= [langs.get(c,c)]
		#return codes,names






	#def extractGoodLanguageParags(self, text, goodLanguageCode):
		#newtext=""
		#for sentence in text.split("\n"):
			#if self.guessLanguage(sentence)[1]==goodLanguageCode:
				#newtext+=sentence+"\n"
		#return newtext
	################################################# end of class Ngram #############################"


from bisect import bisect_left

def binary_search(a, x, lo=0, hi=None):   # can't use a to specify default for hi
	hi = hi if hi is not None else len(a) # hi defaults to len(a)
	pos = bisect_left(a,x,lo,hi)          # find insertion position
	return (pos if pos != hi and a[pos] == x else -1) # don't walk off the end


def serveur(ngram):
	while True:
		t = input("?")
		if t:
			print(ngram.guessLanguage(t))
		else:
			break

def assertions(ngram):
	testset= [
	("ar","تاگلديت ن لمغرب"),
	("fa","برای دیدن موارد مربوط به گذشته صفحهٔ بایگانی را ببینید."),
	("ps","""، کړ و وړو او نورو راز راز پژني او رواني اکرو بکرو... څرګندويي کوي او د چاپېريال او مهال څيزونه، ښکارندې، پېښې،ښه او بد... رااخلي. په بله وينا: ژبه د پوهاوي راپوهاوي وسيله ده.د ژبې په مټ خپل اندونه،واندونه (خيالونه)، ولولې، هيلې او غوښتنې سيده يا ناسيده، عمودي يا افقي نورو ته لېږدولای شو. خبرې اترې يې سيده او ليکنه يې ناسيده ډول دی.که بيا يې هممهالو ته لېږدوو، افقي او که راتلونکو پښتونو( نسلونو) ته يې لېږدوو، عمودي بلل کېږي."),"""),
	("en","the"),
	("ja","ウィキペディアはオープンコンテントの百科事典です。基本方針に賛同していただけるなら、誰でも記事を編集したり新しく作成したりできます。詳しくはガイドブックをお読みください。"),
	("el","Το κεντροδεξιό Εθνικό Κόμμα του Τζον Κέι κερδίζει τις εκλογές στη Νέα Ζηλανδία, αποκαθηλώνοντας από την εξουσία το Εργατικό Κόμμα της Έλεν Κλαρκ, έπειτα από εννέα χρόνια."),
	("ru","Голубь предпочитает небольшие, чаще всего необитаемые острова, где отсутствуют хищники. Живёт в джунглях."),
	("bg","За антихитлеристите месец август 1944 година се оказва добър. "),
	("hi","पहले परमाणु को अविभाज्य माना जाता था।"),
	("gl","Aínda que non nega que os asasinatos de civís armenios ocorreran na realidade, o goberno turco non admite que se tratase dun xenocidio, argumentando que as mortes non foron a resulta dun plano de exterminio masivo organizado polo estado otomán, senón que, en troques, foron causadas polas loitas étnicas, as enfermidades e a fame durante o confuso período da I Guerra Mundial. A pesares desta tese, case tódolos estudosos -até algúns turcos- opinan que os feitos encádranse na definición actual de xenocidio. "),
	("de","Was nicht daran liegt, dass das Unternehmen kein Interesse an der Verarbeitung biologisch angebauter Litschis hat. Die Menge an Obst aber, die Bionade inzwischen braucht, gibt es auf dem weltweiten Biomarkt nicht - oder nur zu einem sehr hohen Preis. Im Prinzip gebe es zwar ausreichend Litschis, allerdings werde ein Großteil der Früchte für den Frischobstmarkt angebaut und auch dort gehandelt. Wandelte man dieses Frischobst in Konzentrat um, würde dies zu teuer, sagte ein Geschäftspartner von Bionade gegenüber Foodwatch. scheiße"),
	("mr","भारतीय रेल्वे (संक्षेपः भा. रे.) ही भारताची सरकार-नियंत्रित सार्वजनिक रेल्वेसेवा आहे. भारतीय रेल्वे जगातील सर्वात मोठ्या रेल्वेसेवांपैकी एक आहे. भारतातील रेल्वेमार्गांची एकूण लांबी ६३,१४० कि.मी. "),
	]
	for languagecode,sampletext in testset:
		guessedcode = ngram.guessLanguage(sampletext)
		print("correct:",languagecode,"guessed:",guessedcode)
		assert languagecode==guessedcode


def extractTypicalWords():
	txt = open("testfiles/arz").read()
	w3 = [w for w in re.split(r"\W",txt) if ('3' in w or '7' in w or '9' in w) and len(w)>2]
	#w3 = [w for w in re.split(r"\W",txt) ]
	print(" ".join(sorted([w for w,_ in Counter(w3).most_common(100)])))


if __name__ == "__main__":
	ngram = Ngram(make_new=False)
	# ngram.makeNgrams() # uncomment when redoing ngram computation because train corpora have changed
	#assertions(ngram)
	print("_________")
	print(ngram.guessLanguage("si on fait des phrases très longues, alors la performances devrait augmenter un peu plus"))
	# print(os.listdir())
	# ngram.NgramsToMatrix()
	# serveur(ngram)

	# ngram.filetest()
