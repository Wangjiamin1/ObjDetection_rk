#include "tracker.h"

#include <cmath>

// const double DIST_THRES = 0.00001;
//const double DIST_THRES = 10;
const double DIST_MAX = 0.0;

static int objNum = 0;

static double CalLonLatDist(StObject& obj1, StObject& obj2)
{
    double dx = obj2.longitude - obj1.longitude;
    double dy = obj2.latitude - obj1.latitude;

    return std::sqrt(dx * dx + dy * dy);
}

// static double CalPtDist(StObject& obj1, StObject& obj2)
// {
//     double dx = obj2.x - obj1.x;
//     double dy = obj2.y - obj1.y;

//     return std::sqrt(dx * dx + dy * dy);
// }

static double CalPtDist(StObject& box1, StObject& box2){
    // 计算两个框的相交区域的坐标和尺寸
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.w, box2.x + box2.w);
    int y2 = std::min(box1.y + box1.h, box2.y + box2.h);

    // 计算相交区域的面积
    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);

    // 计算两个框的并集区域的面积
    int unionArea = box1.w * box1.h + box2.w * box2.h - intersectionArea;

    // 计算 IOU 值
    double iou = static_cast<double>(intersectionArea) / unionArea;
    return iou;
}


bool CTracker::IsLost(StObject &obj)
{
    if(obj.age < 5)
    {
        if(obj.lostframe > 1)
            return true;
    }
    else if(obj.age < 20)
    {
        if(obj.lostframe > 3)
            return true;
    }
    else if(obj.age < 50)
    {
        if(obj.lostframe > 5)
            return true;
    }
    else
    {
        if(obj.lostframe > 10)
            return true;
    }

    return false;
}

void CTracker::update(std::vector<StObject>& objs)
{
    memset(m_unMatchedDetObjIdx, -1, sizeof(m_unMatchedDetObjIdx));
    printf("CTracker::update start\n");
    m_astDetList = objs;
    calDistMat();
    hgrMatch();


    printf("CTracker::update postprocess\n");

    for(int i=0;i<m_astTrackerList.size();i++)
    {
        
        if(m_matchedDetObjIdx[i] != -1) //process matched track and det
        {
            m_astTrackerList[i].longitude = m_astDetList[m_matchedDetObjIdx[i]].longitude;
            m_astTrackerList[i].latitude = m_astDetList[m_matchedDetObjIdx[i]].latitude;

            m_astTrackerList[i].x = m_astDetList[m_matchedDetObjIdx[i]].x ;
            m_astTrackerList[i].y = m_astDetList[m_matchedDetObjIdx[i]].y ;
            m_astTrackerList[i].w = m_astDetList[m_matchedDetObjIdx[i]].w ;
            m_astTrackerList[i].h = m_astDetList[m_matchedDetObjIdx[i]].h ;

            m_astTrackerList[i].lostframe = 0;
            m_astTrackerList[i].age++;
        }
        else //process unmatched track
        {
            m_astTrackerList[i].lostframe++;
        }
    }

    //process unmatched det
    for(int i=0;i<m_astDetList.size();i++)
    {
        if(m_unMatchedDetObjIdx[i] == 1)
        {
            m_astTrackerList.push_back(m_astDetList[i]);
            m_astTrackerList.back().objId = objNum++;
            m_astTrackerList.back().lostframe = 0;
            m_astTrackerList.back().age++;
        }
    }

    //remove lost track
    for (size_t i = 0; i < m_astTrackerList.size();)
    {
        if(IsLost(m_astTrackerList[i]))
        {
            printf("remove obj:%d\n",i);
            //std::cout << "Remove: " << m_tracks[i]->GetID().ID2Str() << ": skipped = " << m_tracks[i]->SkippedFrames() << ", out of frame " << m_tracks[i]->IsOutOfTheFrame() << std::endl;
            m_astTrackerList.erase(m_astTrackerList.begin() + i);
            // assignment.erase(assignment.begin() + i);
        }
        else
        {
            ++i;
        }
    }


    printf("tracker update end,track size:%d\n", m_astTrackerList.size());
    for(auto& track:m_astTrackerList)
    {
        printf("tarck id:%d, clsId:%d, x:%d, age:%d, lostcnt:%d\n", track.objId, track.clsId, track.x, track.age, track.lostframe);
    }


}

double CTracker::calDist(StObject& det, StObject& track)
{
    
    if(det.clsId != track.clsId)
        return DIST_MAX;
    else
    {
        double dist = CalPtDist(det, track);
        return dist;
    }
}

int CTracker::calDistMat()
{
    printf("CTracker::calDistMat start\n");
    int detNum = m_astDetList.size();
    int trackNum = m_astTrackerList.size();
    memset(m_distMat, 0, sizeof(m_distMat));
    for(int i=0;i<detNum;i++)
    {
        for(int j=0;j<trackNum;j++)
        {
            m_distMat[i][j] = calDist(m_astDetList[i], m_astTrackerList[j]);
            printf("%f, ", m_distMat[i][j]);
        }
        printf("\n");
    }
    return 0;

}

bool CTracker::match(int detObjIdx)
{
    for(int i=0;i<m_astTrackerList.size();i++)
    {
        if(m_distMat[detObjIdx][i] > DIST_THRESH && !m_visited[i])
        {
            m_visited[i] = true;
            if(m_matchedDetObjIdx[i] == -1 || match(m_matchedDetObjIdx[i]))
            {
                m_matchedDetObjIdx[i] = detObjIdx;
                return true;
            }
        }
    }
    return false;
}

int CTracker::hgrMatch()
{
    int cnt = 0;
    memset(m_matchedDetObjIdx, -1, sizeof(m_matchedDetObjIdx));
    memset(m_unMatchedDetObjIdx, -1, sizeof(m_unMatchedDetObjIdx));
    for(int i=0;i<m_astDetList.size();i++)
    {
        memset(m_visited, 0, sizeof(m_visited));
        if(match(i))    //match succedd
        {
            cnt++;
        }    
        else    //match failed
        {
            m_unMatchedDetObjIdx[i] = 1;
        }
    }

    return cnt;
}

void CTracker::GetTracks(std::vector<StObject>& tracks)
{
    tracks = m_astTrackerList;
}
