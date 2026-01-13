#pragma once
/**
 *
 * @file untar.h
 * @brief Untar aims to simplify the extractaction of tar files into a stream in
 * memory.
 * @author Alexis Paques
 * @version 1.0.0
 * @date 22 august 2016
 * @description Untar is based on ctar
 * https://github.com/libarchive/libarchive/blob/master/contrib/untar.c, C
 * implementation by libarchive and tarlib
 * http://www.codeproject.com/Articles/470999/tarlib-Windows-TAR-Library which
 * did not compile for me and the libarchive C flat implementation.
 *
 */

#include <fstream>
#include <ios>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>

namespace untar
{

  // Type of file in the TAR documentation
  enum tarEntryType
  {
    FileCompatibilityType = '\0',
    FileType = '0',
    HardlinkType = '1',
    SymlinkType,
    CharacterDeviceType,
    BlockDeviceType,
    DirType,
    FifoType,
    ReservedType,
    OtherType
  };

  // Limit entries to Files or Dir to limit the map size
  enum tarMode
  {
    File = 1,
    Hardlink = 2,
    Symlink = 4,
    CharacterDevice = 8,
    BlockDevice = 16,
    Dir = 32,
    Fifo = 64,
    Reserved = 128,
    All = 255
  };

  // The tarEntry class represents an ENTRY inside a TAR file. It can be a File,
  // Dir, Symlink, ... Important parts to be able to use it :
  // tarEntry::tarEntry(std::string filename, int filesize, std::size_t
  // startOfFile, tarEntryType type, std::string parentTarFilename, std::ifstream
  // * _tarfile); std::ifstream * tarEntry::wantToExtract(int * filesize,
  // std::size_t * startInMemory); std::string getFilename(); std::size_t
  // getStartingByte();
  class tarEntry
  {
    friend class tarFile;

  public:
    // To be able to create null tarEntry
    tarEntry() = default;
    // Default constructor, prefer this one
    tarEntry(std::string filename, int filesize, std::size_t startOfFile,
             tarEntryType type, std::string parentTarFilename,
             std::ifstream *_tarfile);
    // Constructor to avoid error for people who failed the instantiation. Could
    // disapear when I will have properly documented the library.
    tarEntry(tarEntry const &cpy);
    // Destructor
    ~tarEntry() = default;

    // Get the tar filename where this file comes from
    std::string getParentFilename();
    // Get the file size
    int getFileSize();
    // Get the filename (containing the path)
    std::string getFilename();
    // Get the starting byte in the stream
    std::size_t getStartingByte();
    // Get the tar type. Is it a DirType, FileType, SymlinkType ?
    tarEntryType getType();
    // Get all usefull data we need to extract our data in one call
    std::ifstream *wantToExtract(int *filesize, std::size_t *startInMemory);

  private:
    // The stream, this is the same as the one of the tarFile object. Opened by
    // default.
    std::ifstream *_tarfile;
    // If the file is extacted, set to true.
    // For future usage, in my project in SFML
    bool _extracted;
    // Starting byte in the tar file (stream)
    std::size_t _startOfFile;
    // Filesize
    int _filesize;
    // Filename (containing the path)
    std::string _filename;
    // Type of the entry
    tarEntryType _type;
    // What is my dad TAR ?
    std::string _parentTarFilename;
  };

  // tarFile represents a TAR FILE opened. Important parts to be able to use it :
  // tarFile::tarFile(char * filename, int filter = All);
  // static map<std::string, tarEntry *> tarFile::entries
  class tarFile
  {
    friend class tarEntry;

  public:
    // If you don't give a filename, don't forget to initiate via
    // tarFile::open(...)
    tarFile() = default;
    // Default initiation
    tarFile(char *filename, int filter = All);
    // The destructor
    ~tarFile() { close(); }

    // Wrapping the map.find(std::string). Returns the tarEntry, or Null on error;
    tarEntry *find(std::string filename);
    // Wrapping the map.find(std::string). Returns the stream of file, filesize
    // and start bit
    std::ifstream *find(std::string filename, int *filesize, std::size_t *start);

    // Open a file in case you didn't instanciated the class with a filename
    void open(char *filename, int filter = All);
    void close();
    // Get the filename of the opened file
    std::string getFilename();

    std::map<std::string, tarEntry *> getEntries() { return entries; }

  private:
    // Read the file and store entries
    void getAllEntries(int filter = All);
    // Add a new tarEntry into the map (entries)
    void addEntryToMap(std::string filename, int filesize, tarEntryType type);
    // Check if the header is Null. The tar file ends on two Null headers
    bool isNullHeader(const char *p);
    // Verify the checksum, actually, it stops the reading on error
    static int verifyChecksum(const char *p);
    // Used to read the header
    static int parseoct(const char *p, std::size_t n);
    // Did we get entries ? If yes, don't get them again
    bool _get_all_entries;
    // Remember where comes the data form
    char *_filename;

    // Map of tarEntries, containing all the files
    std::map<std::string, tarEntry *> entries;
    // This is the tar filestream
    std::ifstream _tarfile;
  };

} // namespace untar
