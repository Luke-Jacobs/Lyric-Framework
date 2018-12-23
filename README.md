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

Before you can analyze lyrics, you need to have them downloaded into a Collection. Collections are the highest level object in this framework. They encapsulate Artists. Artists encapsule Album objects. Album objects encapsulate Songs. Collections, Artists, and Albums all inherit from CollectionBase. CollectionBase outlines what a collection of Songs look like. To download to a collection, all you need is an artist name and the genre of that artist.

```

```

