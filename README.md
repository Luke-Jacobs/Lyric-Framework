# MultiGenre Lyric Analysis

This program is a framework for collecting and analyzing lyrics.

## Structure

- Song
  - title (str)
  - lyrics (str)
  - methods for analysis (vocabulary and word counts)
- CollectionBase
  - Album - contain Songs
  - Artist - contain Albums
  - Genre - contain Artists
  - Collection - contain Genres

## Downloading lyrics

Before you can analyze lyrics, you need to have them downloaded into a Collection. Collections are the highest level object in this framework. They encapsulate Artists. Artists encapsule Album objects. Album objects encapsulate Songs. Collections, Artists, and Albums all inherit from CollectionBase. CollectionBase outlines what a collection of Songs look like. To download to a collection, all you need is an artist name and the genre of that artist. Then, the framework can download every album produced by that artist from AZlyrics.com. The process takes a long time, so I usually download overnight.

```
usage: LyricFramework.py download [-h] [-o OUTPUT]
                                                      [-a APPEND] [-i INPUT]
                                                      [-d DELAY]

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        The file to store the downloaded lyric data
  -a APPEND, --append APPEND
                        The file to append the new downloaded data to
  -i INPUT, --input INPUT
                        The file that contains a list of artists to download
  -d DELAY, --delay DELAY
                        Delay of the downloader for each song
```

Artist lists are in this format: 

```
Artist A - Genre of A
Artist B - Genre of B
...
```

After you write up a list of artists that you want to download, you can have the framework download those artists using this command:

```
python LyricFramework.py download -i [Artist list path] -o [Output filename for Collection] -d [Delay in seconds for downloading each song]
```

Make sure to set a delay of at least 8 seconds out of consideration for the servers that host AZlyrics.com. After the command is complete, the program should have scraped every lyric written by the artists on your list! Then you can move on to analysis.

## Ranking data

The framework can rank a group of albums, artists, or genres by word frequencies. For example, I ranked all the artists in my Collection by the frequency of self-referencing pronoun usage. So the more an artist uses the wordset ["I", "me", "my", "us", "we"], the higher on the list they will be. Here are my results:

0: Skillet - 0.05974624416109077
1: Tenth Avenue North - 0.05897611963073167
2: Fireflight - 0.056988961298044954
3: Hillsong United - 0.055649751725808404
4: Ledger - 0.05546492659053834
5: Adele - 0.05535703090715316
6: Aaron Cole - 0.05328582083367863
7: Disciple - 0.052732658442108005
8: Ed Sheeran - 0.05227321555471087
9: David Crowder Band - 0.051805237732011185
10: Nine Lashes - 0.051383781691503774
11: Thousand Foot Krutch - 0.050532530678397775
12: Chris Tomlin - 0.049505644765444216
13: Tobymac - 0.048935649620748714
14: Red - 0.04891671269870496
15: Rihanna - 0.048793984570814755
16: Trip Lee - 0.04864018709215043
17: Social Club Misfits - 0.048564862583247354
18: Letter Black - 0.04809560823940925
19: Wolves At The Gate - 0.04753473168074094
20: Ashes Remain - 0.04677914110429448
21: We As Human - 0.046126401630988786
22: Justin Bieber - 0.045850733854338147
23: Memphis May Fire - 0.045748464274167476
24: NF - 0.045202644054216465
25: Lauren Daigle - 0.044606553456110976
26: KB - 0.044504250221536645
27: Casting Crowns - 0.04415291808997747
28: Decyfer Down - 0.04301708898055392
29: Lecrae - 0.042814639905549
30: Tedashii - 0.04252167888965863
31: Andy Mineo - 0.042098276720316685
32: Derek Minor - 0.042022366399180465
33: Lady Gaga - 0.04185427752762996
34: Demon Hunter - 0.0413989720526823
35: Madonna - 0.0409186460874355
36: Pillar - 0.04020869959734589
37: Maroon 5 - 0.03998570957458503
38: Flyleaf - 0.03978310186465999
39: Katy Perry - 0.03974487243621811
40: Newsboys - 0.03861072379262264
41: Flame - 0.03737278322871595
42: Lacey Sturm - 0.037299210454669204
43: Taylor Swift - 0.03666388021177057
44: For King & Country - 0.03612435372322568
45: Matthew West - 0.033403504426970466
