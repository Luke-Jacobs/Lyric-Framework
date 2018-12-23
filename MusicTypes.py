from collections import Counter
from bs4 import BeautifulSoup
import pickle, os.path, re, requests, time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import argparse
from abc import ABC, abstractmethod


class Utilities:

    @staticmethod
    def urlify(s: str) -> str:
        """Take a normal string and convert it to an AZlyrics url format"""
        s = s.lower().replace(" ", "")
        regex = re.compile('[^a-z0-9]')
        s = regex.sub("", s)
        return s

    @staticmethod
    def removeBadCharacters(s: str) -> str:
        """Remove characters that are outside the range of ASCII"""
        byteStr = [ord(char) for char in s if ord(char) < 128]
        return bytes(byteStr).decode()

    @staticmethod
    # get files with extension in a certain directory
    # files are relative paths
    def getFilesWithExtOnPath(path: str, targetExt: str):
        """Get a list of paths to files with a specified extension on a specified path"""
        fileList = []
        for subdir, dirs, files in os.walk(path):
            for name in files:
                ext = os.path.splitext(name)[1]  # get extension to filter files
                if ext == targetExt:
                    fileList.append(os.path.join(subdir, name))
        return fileList

    @staticmethod
    def writeIterable(obj, filename):
        """Write an iterable to a file in a visually-appealing way."""
        with open(filename, "w") as fp:
            for item in obj:
                fp.write(str(item) + "\n")


class Song:
    GENERIC_SONG_URL = 'https://www.azlyrics.com/lyrics/%s/%s.html'  # To be filled by methods
    FAILED_DOWNLOAD_TAG = "NaN"  # self.lyrics is set to this value if the lyrics cannot be downloaded from AZlyrics

    def __init__(self):
        self.title = ""  # Song title
        self.lyrics = ""  # Contents of lyrics in string form
        self.wordCount = None  # Counter object that will store word frequencies

        self.artist = ""
        self.album = ""

    def setLyrics(self, title, lyrics):
        self.title = title
        self.lyrics = lyrics

    def __str__(self):
        """Returns the lyric content of this song."""
        return self.lyrics

    def getDescription(self):
        return "%s by %s from %s" % (self.title, self.artist, self.album)

    def getWordCount(self):
        """Calculates word frequency and sets self.word_freq to a Counter object"""
        if not self.wordCount:
            words = Counter()
            words.update(self.lyrics.split())
            self.wordCount = words
        return self.wordCount

    def getVocabulary(self):
        """Calculates unique words used in lyrics"""
        return set(self.getWordCount())

    def download(self, artist: str, show=True, override=False) -> bool:
        """Given artist and title, download the lyrics of this song from AZlyrics."""

        if not (self.title and artist):  # Throw error if one of these is undefined
            raise RuntimeError("Song: Downloading without knowledge of song title or artist")

        if self.lyrics and not override:  # If no caller override and we already have lyrics downloaded in the object
            return False  # No need to download if we already have lyrics

        titlepath = Utilities.urlify(self.title)
        artistpath = Utilities.urlify(artist)

        url = Song.GENERIC_SONG_URL % (artistpath, titlepath)

        if show:
            print("Downloading (%s)" % url)

        page = requests.get(url)
        if page.status_code == 200:
            self.setLyricsFromHTML(page.text)
            return True
        else:
            self.lyrics = "NaN"  # Constant to use if something went wrong
            return False

    @staticmethod
    def restoreFromPath(path):
        """Retrieve a song object from a local file indicated by a path."""
        with open(path, "rb") as fp:
            song = pickle.load(fp)
        return song

    @staticmethod
    def getSongFromURL(artist_url, title_url):
        """Fills out the self.lyrics variable given self.artist and self.title"""
        # Url follows this pattern: https://www.azlyrics.com/lyrics/[artist urlname]/[title].html
        song = Song()
        url = Song.GENERIC_SONG_URL % (artist_url, title_url)
        page = requests.get(url)
        if page.status_code == 200:
            song.setLyricsFromHTML(page.text)
            return song
        else:
            raise RuntimeError("Downloading Song Page: Status Code not 200")

    @staticmethod
    def cleanLyrics(lyrics: str) -> str:
        """Remove characters and phrases that we don't want to include in our dataset."""
        badCharacters = ["(", ")", "\'", "\'", ","]
        separationCharacters = ["\n\r", "\r\n", "  "]
        for char in badCharacters:
            lyrics = lyrics.replace(char, "")
        for char in separationCharacters:
            lyrics = lyrics.replace(char, " ")
        lyrics = Utilities.removeBadCharacters(lyrics)
        lyrics = lyrics.lower()

        # Manipulating chorus indicators in lyric data
        chorRef = "[chorus]"  # A constant that indicates that a chorus is referenced
        chorDef = "[chorus:]"  # A chorus definition in the AZlyrics format
        if lyrics.find(chorRef) + 1:  # If we found a reference
            if lyrics.find(chorDef) + 1:  # If we found a definition
                start = lyrics.find(chorDef) + len(chorDef)
                end = lyrics[start:].find("\n\n") + start
                chorStr = lyrics[start:end]  # This is the string that holds our chorus
            else:
                print("[-] Chorus reference but no chorus definition")
                chorStr = ""  # If we did not find a chorus
            lyrics = lyrics.replace(chorRef, chorStr)
        lyrics = re.sub(r'\[[^\]]*\]', '', lyrics)

        lyrics = re.sub(r'[?!@#$%^&*().]', '', lyrics)
        return lyrics

    @staticmethod
    def __cleanBracketExpressions(lyrics: str) -> str:
        chorRef = "[chorus]"
        chorDef = "[chorus:]"
        if lyrics.find(chorRef) + 1:  # if reference
            if lyrics.find(chorDef) + 1:  # if definition
                start = lyrics.find(chorDef) + len(chorDef)
                end = lyrics[start:].find("\n\n") + start
                chorStr = lyrics[start:end]
                # print("Found chorus: %s" % chorStr)
            else:
                print("[-] Chorus ref but no def")
                chorStr = ""
            lyrics = lyrics.replace(chorRef, chorStr)

        lyrics = re.sub(r'\[[^\]]*\]', '', lyrics)
        return lyrics

    def setLyricsFromHTML(self, page: str):
        """Set self.lyrics to lyrics from an azlyrics html string"""
        # Get lyrics section from website string
        doc = BeautifulSoup(page, 'html.parser')
        lyrics = doc.find("body")
        lyrics = lyrics.find("div", attrs={"class": "container main-page"})
        lyrics = lyrics.find("div", attrs={"class": "row"})
        lyrics = lyrics.find("div", attrs={"class": "col-xs-12 col-lg-8 text-center"})
        lyrics = lyrics.findAll("div")[6].text
        # Clean lyrics for analysis
        lyrics = Song.cleanLyrics(lyrics)
        self.lyrics = lyrics

    def save(self):
        if not (self.artist and self.album and self.title):
            raise RuntimeError("Lyric saving: artist/album/title not initialized")

        path = os.path.join("Saved Song", self.artist, self.album, self.title) + ".song"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb+") as fp:
            pickle.dump(self, fp)


class LyricCollection:
    """An abstract base class to use for Album, Artist, Genre, and Collection classes.
    Using this as a framework for those classes makes the code look significantly cleaner."""

    def __init__(self):
        self.units = []  # Units are either Songs, Albums, Artists, or Genres.
        self.wordCount = None

    @staticmethod
    def restore(filename: str):
        """Return a CollectionBase object from a file."""
        with open(filename, "rb") as fp:
            obj = pickle.load(fp)
        return obj

    def getSongs(self) -> list:
        if not self.units:  # If this collection contains no data
            return []

        if type(self.units[0]) == Song:  # If this collection is an Album
            return self.units  # Return the Songs directly

        totalSongs = []
        for unit in self.units:
            totalSongs += unit.getSongs()
        return totalSongs  # Return Songs recursively

    def getWordCount(self) -> Counter:
        """Return a Counter object that contains word frequencies."""
        self.wordCount = Counter()
        for unit in self.units:
            self.wordCount += unit.getWordCount()
        return self.wordCount

    def getVocabulary(self) -> set:
        """Return the set of words that this Album uses."""
        vocab = set()
        for unit in self.units:
            vocab.update(unit.getVocabulary())
        return vocab

    def download(self, delay=8):
        """Download the lyrics to every unit in this collection.
        Albums will have to override this function."""
        for unit in self.units:
            unit.downloadAll(delay)

    def save(self, filename: str):
        with open(filename, 'wb') as fp:
            pickle.dump(self, fp)


class Album:

    def __init__(self):
        self.name = ""  # Album title
        self.artist = ""  # Artist name
        self.year = 0  # Album year
        self.songs = []  # List of album Songs
        self.wordCount = None

    def __str__(self):
        return self.name

    def getSongFromTitle(self, songTitle: str) -> Song:
        """Return a Song object given a song's title."""
        for song in self.songs:
            if song.title == songTitle:
                return song
        return None

    def getWordCount(self) -> Counter:
        """Return a Counter object from this Album."""
        self.wordCount = Counter()
        for song in self.songs:
            self.wordCount += song.getWordCount()
        return self.wordCount

    def getWordsetFrequency(self, wordset: list) -> float:
        """Given a list of words (str), return the frequency of these words in this album."""
        freq = 0.0
        wc = self.getWordCount()
        for word in wordset:
            freq += wc[word]  # Add up occurences of each word
        totalOccurences = sum(list(wc.values()))
        if totalOccurences > 0:
            freq /= totalOccurences  # Divide occurences by total words
            return freq
        else:
            return 0.0

    def getVocabulary(self) -> set:
        """Return the set of words that this Album uses."""
        vocab = set()
        for song in self.songs:
            vocab.update(song.getVocabulary())
        return vocab

    def getSongs(self) -> list:
        """Return this album's songs."""
        return self.songs

    def getNsongs(self) -> int:
        """Return the number of songs in this album."""
        return len(self.songs)

    def downloadAll(self, delay=8) -> None:
        """Download the lyrics to every song in this Album."""
        for song in self.songs:
            if not song.download(self.artist):  # Download song's lyrics
                print("Already downloaded song lyrics")
                continue  # Move on if that song is already downloaded
            else:
                time.sleep(delay)


class Artist:
    """Artist type includes Albums and an artist name.
    Its main use is to creates Song types given an artist URL.
    It maintains a collection of Album objects and performs a list of operations on them.
    """

    def __init__(self):
        self.name = ""  # Artist name
        self.albums = []  # List of Albums
        self.count = None  # For word count

    def __str__(self):
        """For an easy description of this Artist object when printing."""
        return self.name

    @staticmethod
    def fromName(name, show=True):
        # To URL string form
        artistPath = Utilities.urlify(name)

        # Get artist main page
        artistURL = "https://www.azlyrics.com/" + artistPath[0] + "/" + artistPath + ".html"
        if show:
            print("Downloading (%s)" % artistURL)
        artistPage = requests.get(artistURL)
        if artistPage.status_code != 200:
            raise RuntimeError("Artist: failed to grab artist page (not 200 status)")

        doc = BeautifulSoup(artistPage.text, 'html.parser')
        albumHTML = doc.find("body")
        albumHTML = albumHTML.find("div", attrs={"class": "container main-page"})
        albumHTML = albumHTML.find("div", attrs={"class": "row"})
        albumHTML = albumHTML.find("div", attrs={"class": "col-xs-12 col-md-6 text-center"})
        albumHTML = albumHTML.find("div", id="listAlbum")  # section with album titles and songs

        albumProperties = albumHTML.findAll("div", attrs={"class": "album"})
        albums = []  # THis will be a list of Albums (name, year, songs)

        # Populate "albums" with Album objects without songs filled out
        for albumProperty in albumProperties:
            if albumProperty.text.find("other songs") + 1:
                albumText = "album: \"Other\" (0)"
            else:
                albumText = albumProperty.text

            albumName = re.search("\"(.*)\"", albumText)
            albumName = albumName.group(1)

            year = re.search("\((.*)\)", albumText)
            year = year.group(1)

            newAlbum = Album()
            newAlbum.name = albumName
            newAlbum.artist = name
            try:
                newAlbum.year = int(year)
            except ValueError:
                newAlbum.year = 0

            albums.append(newAlbum)

        # Match songs with albums
        albumIDsAndSongLinks = albumHTML.findAll("a")
        albumN = -1  # assume album id is the 1st item
        for idOrLink in albumIDsAndSongLinks:
            if idOrLink.attrs.get("id"):  # if album indicator
                albumN += 1  # Move on to the next album
            elif idOrLink.attrs.get("href"):  # if song indicator
                newSong = Song()
                newSong.title = idOrLink.text
                albums[albumN].songs.append(newSong)  # append Song object to specific album
            else:
                raise RuntimeError("Artist: Song page is formatted incorrectly")

        # Build final object
        output = Artist()
        output.name = name
        output.albums = albums

        return output  # Return Artist object

    @staticmethod
    def restore(filename):
        """Restore an Artist object from a pickle file."""
        with open(filename, "rb") as fp:
            obj = pickle.load(fp)
        return obj

    def getAlbumFromTitle(self, albumTitle: str) -> Album:
        """Return an album given its title."""
        for album in self.albums:
            if album.title == albumTitle:  # If we found an album that has the title we are looking for
                return album
        return None  # No albums were found with that title

    def getWordCount(self) -> Counter:
        """Return the Counter object made from the word's in this Artist's albums."""
        if self.count:
            return self.count
        self.count = Counter()
        for album in self.albums:
            self.count += album.getWordCount()
        return self.count

    def getWordsetFrequency(self, wordset):
        """Get frequency of a given wordset in this object's lyric collection."""
        freq = 0.0
        wc = self.getWordCount()
        for word in wordset:
            freq += wc[word]  # Add up occurences of each word
        freq /= sum(list(wc.values()))  # Divide by total words
        return freq

    def getVocabulary(self):
        """Return the total collective vocabulary of this Artist."""
        vocab = set()
        for album in self.albums:
            vocab.update(album.getVocabulary())
        return vocab

    def getSongs(self) -> list:
        """Return all the songs that this artist has made."""
        totalSongs = []
        for album in self.albums:
            totalSongs += album.getSongs()
        return totalSongs

    def getNsongs(self) -> int:
        """Get the total number of songs produced by this artist."""
        albums = [album.getNsongs() for album in self.albums]
        return sum(albums)

    def getAlbums(self) -> list:
        """Return this Artist's Album objects."""
        return self.albums

    def downloadAll(self, delay=8) -> None:
        """Download all the albums associated with this artist."""
        for album in self.albums:
            album.downloadAll(delay)

    def save(self, filename) -> None:
        """Save Artist to a pickle file."""
        with open(filename, "wb+") as fp:
            pickle.dump(self, fp)


class Genre:

    def __init__(self, title="", artists=None):
        self.artists = artists
        self.title = title
        self.wordCount = None

    def __str__(self):
        return self.title

    def __lt__(self, other):
        """For combining genres with the same name - symbol <."""
        if self.title != other.title:
            raise Warning("Title mismatch when combining (%s) and (%s)" % (self.title, other.title))
        self.artists += other.artists

    @staticmethod
    def fromArtistList(artistList: list, delay=5):
        artistObjects = []
        for artist in artistList:
            artistObjects.append(Artist.fromName(artist))
            time.sleep(delay)
        obj = Genre()
        obj.artists = artistObjects
        return obj

    @staticmethod
    def fromRawHTML(htmlFolder: str):
        # Object we are building
        out = Genre()
        # Get files with .raw ending and iterate through them
        lyricFiles = Utilities.getFilesWithExtOnPath(htmlFolder, ".raw")
        for filename in lyricFiles:
            # Get elements in filename
            artist, albumStr, songStr = filename.split("\\")[1::]

            # Get artist object if initialized
            artistObj = out.getArtistsFromNames(artist)

            if not artistObj:
                artistObj = Artist()
                artistObj.name = artist
                out.artists.append(artistObj)

            # Get album object if initialized
            albumName = ' '.join(albumStr.split(" ")[:-1:])  # remove (year) from the end of the folder name
            albumObj = artistObj.getAlbumFromTitle(albumName)

            if not albumObj:
                year = re.search("\((.*)\)", albumStr)
                year = int(year.group(1))
                albumObj = Album()
                albumObj.name = albumName
                albumObj.artist = artist
                albumObj.year = year
                artistObj.albums.append(albumObj)

            # Make new song object
            newSong = Song()
            songTitle = os.path.splitext(songStr)[0]
            newSong.title = songTitle

            # Open and read the raw HTML file
            with open(filename, "rb") as fp:
                rawcontent = fp.read()
            lyricContent = bytes([char for char in rawcontent if char < 128]).decode()
            newSong.setLyricsFromHTML(lyricContent)
            albumObj.songs.append(newSong) #add Song
        return out

    @staticmethod
    def restore(filename: str):
        with open(filename, "rb") as fp:
            obj = pickle.load(fp)
        return obj

    @staticmethod
    def downloadOverTime(title: str, artistList: list, filename: str):
        # Get obj initialized
        if os.path.exists(filename + " - Error"):
            workingGenre = Genre.restore(filename + " - Error")
        else:
            workingGenre = Genre.fromArtistList(artistList)
            workingGenre.title = title
        # Download
        try:
            workingGenre.downloadAll()
        except Exception as e:
            print("Error: %s" % str(e))
            workingGenre.save(filename + " - Error")
            raise e
        workingGenre.save(filename)
        return True

    def getArtistsFromNames(self, artistNames: list) -> list:
        """Retrieves Artist object(s) given their name(s)."""
        out = []  # Will contain our list of Artists
        for artist in self.artists:
            if artist.name in artistNames:
                out.append(artist)
        return out

    def getWordCount(self) -> Counter:
        """Return a Counter object containing this Genre's word frequencies."""
        if self.wordCount:
            return self.wordCount

        self.wordCount = Counter()
        for artist in self.artists:
            self.wordCount += artist.getWordCount()
        return self.wordCount

    def getWordsetFrequency(self, wordset):
        """Get frequency of a given wordset in this object's lyric collection."""
        freq = 0.0
        wc = self.getWordCount()
        for word in wordset:
            freq += wc[word]  # Add up occurences of each word
        freq /= sum(list(wc.values()))  # Divide by total words
        return freq

    def getVocabulary(self) -> set:
        vocab = set()
        for artist in self.artists:
            vocab.update(artist.getVocabulary())
        return vocab

    def getNsongs(self) -> int:
        artists = [artist.getNsongs() for artist in self.artists]
        return sum(artists)

    def getSongs(self) -> list:
        total = []
        for artist in self.artists:
            total += artist.getSongs()
        return total

    def downloadAll(self, delay=10) -> None:
        for artist in self.artists:
            artist.downloadAll(delay)

    def save(self, filename) -> None:
        with open(filename, "wb+") as fp:
            pickle.dump(self, fp)

    def comparedTo(self, otherGenre, show=True) -> dict:
        """Compare the vocabulary of multiple genres."""
        myVocab = self.getVocabulary()
        otherVocab = otherGenre.getVocabulary()

        similarVocab = myVocab.intersection(otherVocab)
        differentVocab = myVocab.difference(otherVocab)

        if show:
            print("Vocab 1: %d\nVocab 2: %d\n" % (len(myVocab), len(otherVocab)))
            print("Similar vocabulary: %d\nDifferent vocabulary: %d" % (len(similarVocab), len(differentVocab)))

        return {"Same": similarVocab, "Different": differentVocab}


class Collection:
    """
    Used for storing many songs from many genres
    """

    def __init__(self):
        self.genres = []  # List of Genre objects

    def __getstate__(self):
        return {'genres': self.genres}

    @staticmethod
    def fromGenreFiles(genreFilenameList: list, genreTitlesList=None):
        """Pulls together many Genre pickle files into a Collection object."""

        obj = Collection()
        for i, file in enumerate(genreFilenameList):
            g = Genre.restore(file)
            if genreTitlesList:  # If caller defines what to title each genre
                g.title = genreTitlesList[i]
            # In case there are multiple Genre obj files with the same name
            existingGenre = obj.getGenresFromNames(g.title)
            if existingGenre:
                existingGenre < g  # Give g's artists to the genre in the collection
            else:
                obj.genres.append(g)
        return obj

    @staticmethod
    def restore(filename: str):
        """Restore a Collection object from a pickle file."""
        with open(filename, "rb") as fp:
            obj = pickle.load(fp)
        return obj

    def save(self, filename: str) -> None:
        """Save this Collection to a pickle file."""
        with open(filename, "wb") as fp:
            pickle.dump(self, fp)

    def add(self, artist: Artist, genreTitle: str):
        """Adds individual Artist to collection"""
        existingGenre = self.getGenresFromNames(genreTitle)
        if not existingGenre:
            newGenre = Genre(title=genreTitle, artists=[artist])
            self.genres.append(newGenre)
        else:
            existingGenre.artists.append(artist)

    def remove(self, artistName: str):
        """Remove an artist from this Collection."""
        for genre in self.genres:
            artist = genre.getArtistsFromNames(artistName)
            if artist: #if found him/her obj
                genre.artists.remove(artist)

    def downloadArtistsFromList(self, listfile: str) -> None:
        """Downloads many artists and adds them to respective genres."""
        with open(listfile, "r") as fp:
            # For each artist - genreTitle pair
            for line in fp:
                # Download Artist obj
                artistName, genreTitle = line.replace("\n", "").split(" - ")
                artist = Artist.fromName(artistName)
                artist.downloadAll()
                # Add Artist to our collection by creating/appending to a new/existing Genre obj
                self.add(artist, genreTitle)

    def getAllArtists(self):
        """Return a flattened list of all Artists."""
        return [artist for genre in self.genres for artist in genre.artists]

    def getAllSongs(self):
        """Get all the songs from this Collection."""

    def getGenresFromNames(self, names) -> list:
        """Retrieves Genre objects given their titles."""
        out = []
        for genre in self.genres:
            if genre.title in names:
                out.append(genre)
        return out

    def getArtistsFromNames(self, names):
        artists = []
        for genre in self.genres:
            artists += genre.getArtistsFromNames(names)

        return artists


class Analysis:
    """
    Analytical functions to use with Collection objects.
    """

    @staticmethod
    def graphGenresWithPCA(genres: list, xName=None, yName=None,
                           title=None, returnVectors=False):
        """Graph Artists in a scatterplot using PCA to reduce word count dimensions."""
        # TODO Generalize to include songs, albums, artists

        # Order our data
        allArtists = [artist for genre in genres for artist in genre.artists]  # List of Artist objs
        orderedNames = [artist.name for artist in allArtists]  # Artist names in order

        # Collect all artists' vocabulary
        allVocab = set()
        for artist in allArtists:
            voc = artist.getVocabulary()
            allVocab.update(voc)
        allVocab = list(allVocab)  # List of all vocab words among genres: ['A', 'B', ...] -

        # Fancy graph variables for color and a legend
        nodeColor = ['purple']*len(allArtists)  # The default color is purple
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
        legendData = []

        artistsWordVectors = [[None]]*len(allArtists)
        for i in range(len(genres)):
            genreColor = colors[i]
            for artist in genres[i].artists:  # Get word vectors for every artist to process by PCA
                wordVector = []  # Fixed length vector: [7, 6, 34, ...]
                wordCount = artist.getWordCount()
                for vocabWord in allVocab:
                    wordVector.append(wordCount[vocabWord])
                index = orderedNames.index(artist.name)
                artistsWordVectors[index] = wordVector
                nodeColor[index] = genreColor
            legendData.append(mpatches.Patch(color=genreColor, label=genres[i].title))  # Add new genre to legend

        # PCA on artist vectors
        pca = PCA(n_components=2).fit(artistsWordVectors)  # Setup and fit our model
        data2D = pca.transform(artistsWordVectors)  # Use model to reduce long vectors to 2d vectors

        # Graph PCA results
        ax = plt.axes()
        ax.axis('equal')  # Force graph zoom to be proportional - x and y scale are the same
        ax.scatter(data2D[:, 0], data2D[:, 1], c=nodeColor)  # Plot our PCA points with corresponding color data
        ax.legend(handles=legendData)
        for i, name in enumerate(orderedNames):  # Assign
            ax.annotate(name, (data2D[i, 0]+1, data2D[i, 1]+1))
        if xName:  # If caller specifies x-axis title
            ax.set_xlabel(xName)
        if yName:  # If caller specifies y-axis title
            ax.set_ylabel(yName)
        if title:  # If caller specifies graph title
            ax.set_title(title)
        plt.show()

        # Return results
        if returnVectors:  # If the caller wants data to return
            return artistsWordVectors, data2D
        else:
            return None

    @staticmethod
    def graphWithTSNE(lyricObjects: list, colorData=None, xName=None, yName=None,
                      title=None, returnVectors=False):
        """Graph Artists in a scatterplot using tSNE to reduce word count dimensions.
        tSNE uses stochastic techniques as opposed to PCA which uses linear transformations.
        Reference: https://lvdmaaten.github.io/tsne/"""

        # TODO Add automatic coloring

        # Order our data
        if hasattr(lyricObjects[0], 'name'):  # Dynamic naming depending on type
            orderedNames = [lyricObj.name for lyricObj in lyricObjects]
        else:
            orderedNames = [lyricObj.title for lyricObj in lyricObjects]

        # Collect all artists' vocabulary
        allVocab = set()
        for lyricObject in lyricObjects:
            voc = lyricObject.getVocabulary()
            allVocab.update(voc)
        allVocab = list(allVocab)  # List of all vocab words among genres: ['A', 'B', ...] -

        objectWordVectors = [[None]] * len(lyricObjects)
        for i, lyricObject in enumerate(lyricObjects):  # Get word vectors for every artist to process by PCA
            wordVector = []  # Fixed length vector: [7, 6, 34, ...]
            wordCount = lyricObject.getWordCount()
            for vocabWord in allVocab:
                wordVector.append(wordCount[vocabWord])
            objectWordVectors[i] = wordVector

        # PCA on artist vectors
        data2D = TSNE(n_components=2).fit_transform(objectWordVectors)  # Use model to reduce long vectors to 2d vectors

        # Graph PCA results
        ax = plt.axes()
        ax.axis('equal')  # Force graph zoom to be proportional - x and y scale are the same
        ax.scatter(data2D[:, 0], data2D[:, 1], c=colorData)  # Plot our PCA points with corresponding color data
        for i, name in enumerate(orderedNames):  # Assign
            ax.annotate(name, (data2D[i, 0] + 1, data2D[i, 1] + 1))
        if xName:  # If caller specifies x-axis title
            ax.set_xlabel(xName)
        if yName:  # If caller specifies y-axis title
            ax.set_ylabel(yName)
        if title:  # If caller specifies graph title
            ax.set_title(title)
        plt.show()

        # Return results
        if returnVectors:  # If the caller wants data to return
            return objectWordVectors, data2D
        else:
            return None

    @staticmethod
    def graphGenresByWordsets(genres: list, wordsetX: list, wordsetY: list,
                              xName=None, yName=None, showArtistNames=True):
        """Graph artists in a scatterplot with genre-based coloring.
        Each artist's position is determined by their vocabulary's frequency
        of 2 wordsets, for the X and Y axis."""

        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
        artistsLoc = []
        artistsName = []  # list of artist names
        artistsColor = []  # list of node colors
        legendPatches = []  # list for legend

        # Generate our graph data from each Artist object
        for i, genre in enumerate(genres):
            currentColor = colors[i]
            legendPatches.append(mpatches.Patch(color=currentColor, label=genre.title))
            for artist in genre.artists:  # add points to empty lists
                xfreq = artist.getWordsetCount(wordsetX)
                yfreq = artist.getWordsetCount(wordsetY)
                artistsLoc.append((xfreq, yfreq))
                artistsName.append(artist.name)
                artistsColor.append(currentColor)

        # Graph it!
        data = np.array(artistsLoc, dtype=float)
        ax = plt.axes()
        ax.scatter(data[:, 0], data[:, 1], c=artistsColor)
        ax.axis('equal')
        ax.legend(handles=legendPatches)

        # Extra options for graphing
        if showArtistNames:
            for i, name in enumerate(artistsName):
                ax.annotate(name, (data[i, 0], data[i, 1]))
        if xName:
            ax.set_xlabel(xName)
        if yName:
            ax.set_ylabel(yName)

        plt.show()

    @staticmethod
    def graphWordsetOverTime(wordset: list, genres=None, artists=None,
                             xName=None, yName=None, title=None):
        """Show a line graph of the evolution of either genres or artists over time.
        Each point on the graph is an average frequency of the wordset at that moment
        in time."""

        legendData = []
        # Main data collection
        fig, ax = plt.subplots()

        # If the caller wants to graph the evolution of individual *genres* over time
        # TODO Make clearer
        if genres:
            for i, genre in enumerate(genres):
                yearFreqDict = {}  # Years that an album in this genre was released -> freq of wordset

                # Collect a list of Genre objs in ascending order by year
                for artist in genre.artists:
                    for album in artist.albums:
                        if album.year == 0:
                            continue
                        if not album.year in yearFreqDict: #if new year for an album
                            yearFreqDict[album.year] = [album.getWordsetCount(wordset), 1] #the 1 is used for averages
                        yearFreqDict[album.year][0] += album.getWordsetCount(wordset)
                        yearFreqDict[album.year][1] += 1 #add 1 to the final divisor to calculate an avg
                for year in yearFreqDict:
                    yearFreqDict[year] = yearFreqDict[year][0] / yearFreqDict[year][1]  # calculate avg

                # Plot this genre's data
                dataPoints = list(yearFreqDict.items())
                dataPoints.sort(key=lambda x: x[0])  # Sort by year value (x)
                xVals = [item[0] for item in dataPoints]
                yVals = [item[1] for item in dataPoints]
                legendData.append(ax.plot(xVals, yVals, label=genre.title)[0])

        # If the caller wants to graph the evolution of multiple *artists* over time
        if artists:
            for i, artist in enumerate(artists):
                yearFreqDict = {}  # Years that an album by this artist was released -> freq of wordset
                for album in artist.albums:
                    if album.year == 0:
                        continue
                    if album.year not in yearFreqDict:  # if new year for an album
                        yearFreqDict[album.year] = album.getWordsetCount(wordset)
                        continue
                    yearFreqDict[album.year] += album.getWordsetCount(wordset)
                    yearFreqDict[album.year] /= 2.0  # Average value for

                # Plot this genre's data
                dataPoints = list(yearFreqDict.items())
                dataPoints.sort(key=lambda x: x[0])  # sort by year value (x)
                xVals = [item[0] for item in dataPoints]
                yVals = [item[1] for item in dataPoints]
                legendData.append(ax.plot(xVals, yVals, label=artist.name)[0])

        # Graph parameters
        ax.legend(handles=legendData)
        if xName:
            ax.set_xlabel(xName)
        if yName:
            ax.set_ylabel(yName)
        if title:
            ax.set_title(title)
        plt.show()

    @staticmethod
    def rankByWordsetFrequency(lyricObjects: list, wordset: list) -> list:
        """Return a ranking of artists based on the frequency their vocabulary uses the words in the wordset."""
        ranking = []
        for lyricObject in lyricObjects:
            value = lyricObject.getWordsetFrequency(wordset)
            ranking.append((lyricObject, value))  # Organize abstract lyric objects
        ranking.sort(key=lambda x: -x[1])  # Ascending organization for our ranking by the second element (frequency)
        return ranking

    @staticmethod
    def printRanking(ranking: list) -> None:
        """Convenience function to print a ranking in a pretty way."""
        for i, entry in enumerate(ranking):
            print("%d: %s - %s" % (i, str(entry[0]), str(entry[1])))  # Number - Object - Ranking value

    @staticmethod
    def getUniqueVocabOfTarget(target, others: list) -> set:
        """Returns the difference of target's vocab and the collective vocab of others.
        "Target" and "others" can be a Song, Album, Artist, or Genre."""
        targetVocab = target.getVocabulary()
        otherVocab = set()  # The set of the vocabulary from each other
        for other in others:
            otherVocab.update(other.getVocabulary())
        return targetVocab.difference(otherVocab)  # Return unique vocab

    @staticmethod
    def compareArtistToOthers(artist: Artist, others: list) -> dict:
        """Return a dictionary of [other artist]: [vocab similarities] for a given artist."""
        comparisons = {}
        for other in others:
            if other is artist:
                continue
            # Get n of similarities
            selfVoc = artist.getVocabulary()
            otherVoc = other.getVocabulary()
            comparisons[other.name] = len(selfVoc.intersection(otherVoc))
        return comparisons  # Dictionary {artistName: similarity score}


selfPronouns = ["me", "my", "mine"]
collectivePronouns = ["we", "our", "us"]
otherPronouns = ["you", "your", "he", "she", "his", "her", "they", "them"]


def parseInputs():
    INSUFFICIENT_ARGUMENTS = -1

    args = argparse.ArgumentParser(description='A program to download and analyze lyrics')
    args.add_argument('mode', type=str,
                      help='The mode of program function, either download, graph, or rank')

    # Download args
    args.add_argument('-o', '--output', help='The file to store the downloaded lyric data')
    args.add_argument('-i', '--input', help='The file that contains a list of artists to download')
    args.add_argument('-d', '--delay', help='Delay of the downloader for each song')

    # Specification args
    args.add_argument('-c', '--collection', required=False,
                      help='Collection to be an input')
    args.add_argument('-a', '--artists', required=False, nargs='+',
                      help='Artist names in collection to be specified as inputs')
    args.add_argument('-g', '--genres', required=False, nargs='+',
                      help='Genre names in collection to be specified as inputs')

    # Graphing args
    # -Additional Options
    args.add_argument('-pca', required=False, default=False, action="store_true",
                      help='Use Principle Component Analysis to graph artists')
    args.add_argument('-scatter', required=False, default=False, action="store_true",
                      help='Make a scatterplot')
    args.add_argument('-x', required=False, nargs='+',  # Accepts a list of strings (wordset)
                      help='X axis wordset')
    args.add_argument('-y', required=False, nargs='+',  # Same as X wordset
                      help='Y axis wordset')

    # Rank args
    args.add_argument('-w', '--wordset', nargs='+',
                      help='Input wordset used to rank lyric collection')

    # Parse args
    args = args.parse_args()

    if args.mode == 'download':  # If the user wants to download some lyrics
        pass
    elif args.mode == 'graph':  # If the user wants to use stats to graph lyric data
        if args.collection:  # Draw inputs from our repository of lyrics
            collection = Collection.restore(args.collection)  # Load collection from arguments
            if args.genres:
                pass  # TODO
        else:
            print('No collection to draw from')
            exit(INSUFFICIENT_ARGUMENTS)
    elif args.mode == 'rank':  # If the user wants to rank songs/albums/artists/genres on certain criteria
        if args.collection:  # Draw inputs from our repository of lyrics
            collection = Collection.restore(args.collection)  # Load collection from arguments
            if args.genres:  # Specified genres to rank
                rankingObjs = collection.getGenresFromNames(args.genres)
            elif args.artists:  # Specified artists to rank
                rankingObjs = collection.getArtistsFromNames(args.artists)
            else:  # By default, rank all genres
                rankingObjs = collection.genres
            results = Analysis.rankByWordsetFrequency(rankingObjs, args.wordset)
            Analysis.printRanking(results)  # For pretty output
        else:
            print('No collection to draw from')
            exit(INSUFFICIENT_ARGUMENTS)
    else:
        print('Unknown mode')
        return -1


if __name__ == "__main__":
    # parseInputs()

    mc = Collection.restore("Music Collection")
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
    colorArtists = [colors[i] for i in range(len(mc.genres))]
    Analysis.graphWithTSNE(mc.genres,
                           colorData=colorArtists,
                           xName='tSNE X Axis',
                           yName='tSNE Y Axis',
                           title='tSNE Artist Clustering')

    exit(0)
    # mc = Collection.restore("Music Collection - Failed")
    # try:
    #     mc.downloadArtistsFromList("Artist List.txt")
    # except Exception as e:
    #     print(e)
    #     mc.save("Music Collection - Failed")
    #     exit(-1)
    # mc.save("Music Collection - Updated")

    raw, processed = Analysis.graphGenresWithPCA(mc.genres,
                                                 xName='Principle Component 1 (correlated with size)',
                                                 yName='Principle Component 2',
                                                 title='Artist vocabulary clustered by similarity',
                                                 returnVectors=True)

    Utilities.writeIterable(raw, "RawArtistVectors.txt")
    Utilities.writeIterable(processed, "ProcessedPCAVectors.txt")
