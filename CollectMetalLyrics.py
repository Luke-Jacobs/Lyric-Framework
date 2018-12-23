import requests, os, re, time, pickle, matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from collections import Counter
from MusicTypes import Song, Album, Artist, Utilities

musiciansToGet = ['Skillet',
                  'Red',
                  'Nine Lashes',
                  'Fireflight',
                  'Ledger',
                  'Flyleaf',
                  'We As Human',
                  'Disciple',
                  'Ashes Remain',
                  'Demon Hunter']

# ----- Retrieval -----

# Get a song's lyrics from an html page
def getLyricsFromPage(page: str) -> str:
    #Get lyrics section from website string
    doc = BeautifulSoup(page, 'html.parser')
    lyrics = doc.find("body")
    lyrics = lyrics.find("div", attrs={"class": "container main-page"})
    lyrics = lyrics.find("div", attrs={"class": "row"})
    lyrics = lyrics.find("div", attrs={"class": "col-xs-12 col-lg-8 text-center"})
    lyrics = lyrics.findAll("div")[6].text
    #Clean lyrics for analysis
    lyrics = Song.__cleanLyrics(lyrics)
    #print(lyrics)
    return lyrics

#get files with extension in a certain directory
#files are relative paths
def getFilesWithExtOnPath(path: str, targetExt: str):
    fileList = []
    for subdir, dirs, files in os.walk(path):
        for name in files:
            ext = os.path.splitext(name)[1] #get extension to filter files
            if ext == targetExt:
                fileList.append(os.path.join(subdir, name))
    return fileList

#writes lyric files beside raw files
def extractLyricsOnPaths(paths: list):
    for path in paths:
        lyricFiles = getFilesWithExtOnPath(path, ".raw")
        for filename in lyricFiles:
            with open(filename, "rb") as fp:
                rawcontent = fp.read()
            rawcontent = bytes([char for char in rawcontent if char < 128]).decode()
            lyriccontent = getLyricsFromPage(rawcontent)
            writeFile = filename.replace(".raw", ".lyric")
            with open(writeFile, "w") as fp:
                fp.write(lyriccontent)

#write one file of raw html from url
def writeLyricsFromURL(url: str, path: str) -> bool:
    page = requests.get(url)
    if page.status_code == 200:
        #write raw html to file
        path = path.replace("?", "")
        path = Song.__removeBadCharacters(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".raw", "wb+") as fp:
            fp.write(page.text.encode())
        return True
    else:
        return False

#get songs from songlist slowly
def getSongListAtOnce(songlist: list, delay=10):
    for song in songlist:
        if os.path.isfile(song[0] + ".raw"):
            print("[i] Song (%s) from URL (%s) exists already" % (song[1], song[0]))
            break
        success = writeLyricsFromURL(*song)
        if not success:
            print("[-] Error 200 for %s" % song[1])
        else:
            print("[%d] Wrote (%s) from URL (%s)" % (songlist.index(song), song[1], song[0]))
        time.sleep(delay)

#----- Creating song lists to retrieve -----

#get list of songs by one artist
def getSongsByArtist(artistPath: str, artistName: str) -> list:
    #Create artist folder
    try:
        os.mkdir(artistName)
    except FileExistsError:
        print("[i] Artist directory (%s) exists" % artistName)
    os.chdir(artistName)
    #Get artist main page
    artistURL = "https://www.azlyrics.com/" + artistPath[0] + "/" + artistPath + ".html"
    artistPage = requests.get(artistURL)
    if artistPage.status_code != 200:
        return []
    #Parse artist page into list of albums [album, year, title, names [str], urls [str]]
    artistData = []

    doc = BeautifulSoup(artistPage.text, 'html.parser')
    albumHTML = doc.find("body")
    albumHTML = albumHTML.find("div", attrs={"class": "container main-page"})
    albumHTML = albumHTML.find("div", attrs={"class": "row"})
    albumHTML = albumHTML.find("div", attrs={"class": "col-xs-12 col-md-6 text-center"})
    albumHTML = albumHTML.find("div", id="listAlbum") #section with album titles and songs

    albumProperties = albumHTML.findAll("div", attrs={"class": "album"})
    albumTextProperties = [] #list of [name (str), year (str), empty (list)]
    for albumProperty in albumProperties:
        if albumProperty.text.find("other songs") + 1:
            continue #remove this extraneous section

        name = re.search("\"(.*)\"", albumProperty.text)
        name = name.group(1)

        year = re.search("\((.*)\)", albumProperty.text)
        year = year.group(1)

        albumTextProperties.append([name, year, []])

    #match song links with albums
    albumIDsAndSongLinks = albumHTML.findAll("a")

    albumN = -1 #assume album id is the 1st item
    for idOrLink in albumIDsAndSongLinks:
        if idOrLink.attrs.get("id"): #if album indicator
            albumN += 1
        elif idOrLink.attrs.get("href"): #if song indicator
            songName = idOrLink.text
            songURL = idOrLink.attrs["href"]
            albumTextProperties[albumN][2].append([songName, songURL]) #append song name
        else:
            return []
    #at this point, albumTextProperties should be [albums [name, year, songs [name, url]]]
    #make list of URLs and file paths to write to
    songsToGet = [] #[url job [url, path]]
    for album in albumTextProperties:
        albumName = Utilities.removeBadCharacters(album[0])
        yearName = album[1]
        folderName = "%s (%s)" % (albumName, yearName)
        #make folder for album
        try:
            os.mkdir(folderName)
        except FileExistsError:
            print("[i] Directory exists")
        for song in album[2]: #song is [name, url]
            writePath = "%s/%s/%s" % (artistName, folderName, song[0]) #path to write individual song

            urlParts = song[1].split("/")
            if urlParts[:2] != ['..', 'lyrics']:
                print("[-] Error parsing song URL!")
                break
            absoluteURL = 'https://www.azlyrics.com/' + '/'.join(urlParts[1:])

            songsToGet.append([absoluteURL, writePath])
    os.chdir("..")
    return songsToGet #[songs [url, path]]

#get songlists from many artists
def getSongListFromArtists(artists: list, delay=7.0) -> list:
    allSongs = []
    for artist in artists:
        allSongs += getSongsByArtist(*artist)
        time.sleep(delay)
    return allSongs


if __name__ == "__main__":
    skillet = Artist.fromName("Skillet")

    # allMetalSongPaths = getFilesWithExtOnPath("Saved Song", ".song")
    # allMetalLyrics = [Song.restoreFromPath(path) for path in allMetalSongPaths]
    # artists = [artist[1] for artist in musiciansToGet]
    # genreFreqs = {}
    # fp = open("artist_wf", "w")
    # for artist in artists:
    #     genreFreqs[artist] = Counter()
    #     for lyrics in allMetalLyrics:
    #         if lyrics.artist == artist:
    #             genreFreqs[artist] += lyrics.getWordFrequency()
    #
    # for artist in artists:
    #     fp.write(artist + "\n" + "-"*10 + "\n")
    #     ranking = genreFreqs[artist].most_common()
    #
    #     for i in range(len(ranking)):
    #         fp.write("\t%s: %d\n" % (ranking[i][0], ranking[i][1]))
    #     fp.write("\n")
    # fp.close()

