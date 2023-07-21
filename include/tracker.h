#ifndef _TRACKER_H_
#define _TRACKER_H_

#include "common.h"
#include <vector>
#include <cstring>

const int MAX_DET_NUM = 50;
const int MAX_TRACK_NUM = 50;
const float DIST_THRESH = 0.3;

class CTracker
{

public:
    CTracker() = default;
    ~CTracker() = default;

    void update(std::vector<StObject>& objs);
    void GetTracks(std::vector<StObject>& tracks);

private:
    int calDistMat();
    bool match(int detObjIdx);
    int hgrMatch();
    double calDist(StObject& det, StObject& track);
    bool IsLost(StObject &obj);

    std::vector<StObject> m_astTrackerList;
    std::vector<StObject> m_astDetList;
    double m_distMat[MAX_DET_NUM][MAX_TRACK_NUM];
    int m_matchedDetObjIdx[MAX_TRACK_NUM];
    int m_unMatchedDetObjIdx[MAX_TRACK_NUM];
    bool m_visited[MAX_TRACK_NUM];
};

#endif
