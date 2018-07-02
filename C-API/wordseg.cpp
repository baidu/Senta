#include <iostream>
#include "wordseg.h"
using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

namespace senti_cls_dnn {
    void* WordSegment::_s_lac_handle = NULL;
    
    int WordSegment::global_init(const string& config_path) {
        // Global Resources Init for lac
        string wordseg_conf_path = config_path + "/lac/conf/";
        _s_lac_handle = lac_create(
            wordseg_conf_path.c_str());
        if (_s_lac_handle == NULL) {
            cerr << "[ERROR] Init Worddict failed!" << endl;
            return -1;
        }
        return 0;
    }

    void WordSegment::global_destroy() {
        // Global Resources Destroy for lac
        if (_s_lac_handle != NULL) {
            lac_destroy(_s_lac_handle);
            _s_lac_handle = NULL;
        }
    }

    int WordSegment::thread_init() {
        // Thread Resources Init for lac
        _lac_buff = lac_buff_create(
            _s_lac_handle);
        if (_lac_buff == NULL) {
            cerr << "[ERROR] Create lac_buff failed!" << endl;
            return -1;
        }
        _results = new tag_t[SEN_MAX_TOKENS + 1];
        if (_results == NULL) {
            cerr << "[ERROR] Create tag_t failed!" << endl;
            return -1;
        }
        return 0;
    }

    void WordSegment::thread_destroy() {
        // Thread Resources Destroy for lac
        if (_lac_buff != NULL) {
            lac_buff_destroy(_s_lac_handle, 
                _lac_buff);
            _lac_buff = NULL;
        }
        if (_results != NULL) {
            delete [] _results;
            _results = NULL;
        }
    }

    int WordSegment::word_segment(const string& str_input,
        vector<string>& word_list) {
        // do word segment by lac
        if (_s_lac_handle == NULL || _lac_buff == NULL
            || str_input.empty()) {
            cerr << "[ERROR] Failed in word_segment check!" 
                << endl;
            return -1;
        }
        int count = lac_tagging(_s_lac_handle,
            _lac_buff, str_input.c_str(), _results, SEN_MAX_TOKENS);
        // get word segment results
        for (int i = 0; i < count; i++) {
            string word_str = str_input.substr(_results[i].offset, 
                _results[i].length);
            word_list.push_back(word_str);
        }
        if (word_list.size() <= 0) {
            cerr << "[ERROR] Failed in word_segment!" << endl;
            return -1;
        }
        return 0;
    }
}

