#ifndef BPE_HPP
#define BPE_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map> 
#include <map>
#include <utility>
#include <queue>
#include <fstream>

using byte_map = std::unordered_map<uint32_t, std::vector<uint8_t>>;
using freqDist = std::unordered_map<uint64_t, size_t>;
using byte_template = std::vector<std::pair<uint32_t, uint32_t>>;
using template_map = std::unordered_map<uint64_t, uint32_t>;
using encodeMap = std::map<std::pair<uint32_t, uint32_t>, uint32_t>;

class BPE {
private: 
    byte_map vocab; 
    std::vector<uint32_t> buffer; 
    uint64_t nextPair;
    freqDist pairCounts;
    std::priority_queue<std::pair<size_t, uint64_t>> pairQueue; 
    std::unordered_map<uint32_t, std::vector<size_t>> positions;
    byte_template mergeHistory; 
    template_map mergeMap; 
    encodeMap mergeRules; 
    uint32_t LIMIT = 15000;
    size_t minFreq = 30; 
    uint32_t magicNumber = 0x42504531; //hex for BPE1
    const uint32_t WORD_START = 0x110000;
    const uint32_t WORD_END = 0x110001;
    const uint32_t SPACE = 0x110002;
    const uint32_t TAB = 0x110003;
    const uint32_t NEWLINE = 0x110004;
    const uint32_t INVALID_TOKEN = 0xFFFFFFFF;
public:
    BPE() = default; 
    void reset();
    std::vector<uint32_t> preTokenize(const std::string& text);
    std::vector<uint32_t> preTokenizeCode(const std::string& code);
    std::vector<uint32_t> encodeStr(const std::string& text);
    void initVocab(); 
    uint32_t getVocabSize() const;
    void setVocabSize(const size_t limit);
    void setMinFreq(const size_t freq);
    uint64_t getBestPair();
    size_t findNextValidIndex(size_t index);
    size_t findPrevValidIndex(size_t index);
    uint64_t countPairs(std::vector<uint32_t>& buffer);
    void countSubwordPairs(std::vector<uint32_t>& buffer);
    void updateVocab(const uint64_t key);
    byte_map mergePairs(std::vector<uint32_t>& tokenBuffer);
    byte_map fitText(const std::string& text);
    byte_map fitWords(const std::string& text);
    byte_map fitCode(const std::string& code); 
    void updateBuffer(const uint32_t id1, const uint32_t id2, const uint32_t newID); 
    std::vector<uint32_t> tokenize(const std::string& text);
    std::vector<uint32_t> tokenizeWords(const std::string& text);
    std::string decode(const std::vector<uint32_t>& byteBuffer);
    void save(const std::string& filename); 
    void load(const std::string& filename); 
};

#endif


