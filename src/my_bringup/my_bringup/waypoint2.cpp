#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

using namespace std::chrono_literals;
using NavigateToPose = nav2_msgs::action::NavigateToPose;
using GoalHandleNav  = rclcpp_action::ClientGoalHandle<NavigateToPose>;

enum class State { IDLE, MOVING, SCANNING, DETECTED, RETURNING, WAITING_RESPONSE };

struct Waypoint { double x, y, yaw; };

static const std::vector<Waypoint> WAYPOINTS = {
    {  7.53, -19.99, 0.0},
    { 11.00,  -6.86, 0.0},
    { -0.25,  -2.71, 0.0},
    { -0.09,   3.96, 0.0},
    { 11.07,   7.36, 0.0},
    {  0.08,  11.63, 0.0},
    {-10.97,   7.52, 0.0},
    { -8.33,  19.76, 0.0},
    {  8.41,  20.41, 0.0},
};

static const Waypoint HOME = {-12.59, -22.42, 0.0};

static constexpr double FX                   = 381.3611602783203;
static constexpr double FY                   = 381.361141204834;
static constexpr double CX_CAM               = 320.0;
static constexpr double CY_CAM               = 240.0;
static constexpr double CAM_OFFSET_X         = 0.305;
static constexpr double CAM_OFFSET_Y         = 0.010;
static constexpr double SPIN_VEL             = 0.5;
static constexpr double SPIN_DURATION        = (2.0 * M_PI) / SPIN_VEL;
static constexpr double CONFIDENCE_THRESHOLD = 0.75;


class SearchRobot : public rclcpp::Node
{
public:
    SearchRobot() : Node("search_robot"), state_(State::IDLE)
    {
        cmd_pub_    = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        speak_pub_  = create_publisher<std_msgs::msg::String>("speak", 10);
        listen_pub_ = create_publisher<std_msgs::msg::Bool>("/listen_enable", 10);

        img_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&SearchRobot::imageCallback, this, std::placeholders::_1));

        det_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
            "/yolo/detections", 10,
            std::bind(&SearchRobot::detectionCallback, this, std::placeholders::_1));

        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&SearchRobot::scanCallback, this, std::placeholders::_1));

        amcl_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/amcl_pose", 10,
            std::bind(&SearchRobot::amclCallback, this, std::placeholders::_1));

        voice_sub_ = create_subscription<std_msgs::msg::String>(
            "voice_cmd", 10,
            std::bind(&SearchRobot::voiceCallback, this, std::placeholders::_1));

        nav_client_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");

        speak("Search robot ready. Say find followed by an object name to start.");
    }

private:
    State       state_;
    std::string target_;

    std::mutex frame_mutex_;
    cv::Mat    latest_frame_;

    std::mutex det_mutex_;
    bool       object_found_ = false;
    double     object_cx_    = 0.0;
    double     object_cy_    = 0.0;
    int        detect_count_ = 0;

    std::mutex                  scan_mutex_;
    sensor_msgs::msg::LaserScan latest_scan_;
    bool                        scan_valid_ = false;

    std::mutex amcl_mutex_;
    double     robot_x_   = 0.0;
    double     robot_y_   = 0.0;
    double     robot_yaw_ = 0.0;
    bool       amcl_valid_ = false;

    std::mutex              voice_mutex_;
    std::condition_variable voice_cv_;
    std::string             voice_response_;
    bool                    voice_received_ = false;

    std::mutex               nav_mutex_;
    std::condition_variable  nav_cv_;
    bool                     nav_done_    = false;
    bool                     nav_success_ = false;
    GoalHandleNav::SharedPtr current_goal_handle_;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr             cmd_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr                 speak_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr                   listen_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr            img_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr det_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr        scan_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr              voice_sub_;
    rclcpp_action::Client<NavigateToPose>::SharedPtr                    nav_client_;

    // ── Speak ─────────────────────────────────────────────────────────────────
    void speak(const std::string & text)
    {
        RCLCPP_INFO(get_logger(), "%s", text.c_str());
        std_msgs::msg::String msg;
        msg.data = text;
        speak_pub_->publish(msg);
    }

    // ── Callbacks ─────────────────────────────────────────────────────────────
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_ = cv_ptr->image.clone();
        } catch (...) {}
    }

    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(scan_mutex_);
        latest_scan_ = *msg;
        scan_valid_  = true;
    }

    void amclCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(amcl_mutex_);
        robot_x_ = msg->pose.pose.position.x;
        robot_y_ = msg->pose.pose.position.y;
        double qx = msg->pose.pose.orientation.x;
        double qy = msg->pose.pose.orientation.y;
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;
        robot_yaw_ = std::atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz));
        amcl_valid_ = true;
    }

   // REPLACE ONLY THIS FUNCTION IN YOUR CODE

void detectionCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
    if (state_ != State::MOVING && state_ != State::SCANNING) return;
    if (target_.empty()) return;

    std::lock_guard<std::mutex> lock(det_mutex_);
    bool found_this_frame = false;

    for (const auto & det : msg->detections) {
        if (det.results.empty()) continue;

        // 🔥 CHECK ALL RESULTS (important)
        for (const auto & res : det.results) {

            double confidence = res.hypothesis.score;
            if (confidence < CONFIDENCE_THRESHOLD) continue;

            std::string label        = res.hypothesis.class_id;
            std::string label_lower  = label;
            std::string target_lower = target_;

            std::transform(label_lower.begin(),  label_lower.end(),  label_lower.begin(),  ::tolower);
            std::transform(target_lower.begin(), target_lower.end(), target_lower.begin(), ::tolower);

            if (label_lower == target_lower) {
                found_this_frame = true;
                object_cx_ = det.bbox.center.position.x;
                object_cy_ = det.bbox.center.position.y;

                RCLCPP_INFO(get_logger(),
                    "Detection: %s conf=%.2f state=%s count=%d",
                    label.c_str(), confidence,
                    state_ == State::SCANNING ? "SCAN" : "MOVE",
                    detect_count_ + 1);

                break;
            }
        }
    }

    if (found_this_frame) {
        detect_count_++;

        // 🔥 FIX: SAME LOGIC FOR BOTH SCAN + MOVE
        if (detect_count_ >= 3) {
            object_found_ = true;
        }

    } else {
        detect_count_ = 0;
        object_found_ = false;
    }
}

    void voiceCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        std::string text = msg->data;
        std::transform(text.begin(), text.end(), text.begin(), ::tolower);
        text.erase(0, text.find_first_not_of(" \n\r\t"));
        text.erase(text.find_last_not_of(" \n\r\t") + 1);
        if (text.empty()) return;

        if (state_ == State::IDLE) {
            auto pos = text.find("find ");
            if (pos != std::string::npos) {
                target_ = text.substr(pos + 5);
                target_.erase(target_.find_last_not_of(" \n\r\t") + 1);
                if (target_.empty()) return;
                speak("Command received. Finding " + target_);
                std::thread(&SearchRobot::runMission, this).detach();
            }
            return;
        }

        if (state_ == State::WAITING_RESPONSE) {
            std::lock_guard<std::mutex> lock(voice_mutex_);
            voice_response_ = text;
            voice_received_ = true;
            voice_cv_.notify_all();
        }
    }

    // ── LiDAR + Camera fusion ─────────────────────────────────────────────────
    bool getObjectCoordinates(double cx_px, double cy_px,
                              double & obj_map_x, double & obj_map_y)
    {
        sensor_msgs::msg::LaserScan scan;
        double rx, ry, ryaw;
        {
            std::lock_guard<std::mutex> lock(scan_mutex_);
            if (!scan_valid_) return false;
            scan = latest_scan_;
        }
        {
            std::lock_guard<std::mutex> lock(amcl_mutex_);
            if (!amcl_valid_) return false;
            rx = robot_x_; ry = robot_y_; ryaw = robot_yaw_;
        }

        double ray_x           = (cx_px - CX_CAM) / FX;
        double ray_angle_robot = std::atan2(ray_x, 1.0);

        int    num_beams = static_cast<int>(scan.ranges.size());
        double angle_min = scan.angle_min;
        double angle_inc = scan.angle_increment;

        int    best_idx        = -1;
        double best_angle_diff = 1e9;
        for (int i = 0; i < num_beams; ++i) {
            double diff = std::abs((angle_min + i * angle_inc) - ray_angle_robot);
            if (diff < best_angle_diff) { best_angle_diff = diff; best_idx = i; }
        }
        if (best_idx < 0) return false;

        double distance = 0.0;
        int    count    = 0;
        for (int i = std::max(0, best_idx - 2);
             i <= std::min(num_beams - 1, best_idx + 2); ++i)
        {
            double r = scan.ranges[i];
            if (std::isfinite(r) && r > scan.range_min && r < scan.range_max)
                { distance += r; count++; }
        }
        if (count == 0) return false;
        distance /= count;

        double beam_angle  = angle_min + best_idx * angle_inc;
        double obj_robot_x = distance * std::cos(beam_angle) + CAM_OFFSET_X;
        double obj_robot_y = distance * std::sin(beam_angle) + CAM_OFFSET_Y;

        obj_map_x = rx + obj_robot_x * std::cos(ryaw) - obj_robot_y * std::sin(ryaw);
        obj_map_y = ry + obj_robot_x * std::sin(ryaw) + obj_robot_y * std::cos(ryaw);
        return true;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    bool detectObject(double & cx_out, double & cy_out)
    {
        std::lock_guard<std::mutex> lock(det_mutex_);
        if (object_found_) { cx_out = object_cx_; cy_out = object_cy_; return true; }
        return false;
    }

    void resetDetection()
    {
        std::lock_guard<std::mutex> lock(det_mutex_);
        object_found_ = false;
        detect_count_ = 0;
    }

    void stop() { cmd_pub_->publish(geometry_msgs::msg::Twist()); }

    void cancelNav()
    {
        std::lock_guard<std::mutex> lock(nav_mutex_);
        if (current_goal_handle_) {
            nav_client_->async_cancel_goal(current_goal_handle_);
            current_goal_handle_ = nullptr;
        }
    }

    void setListening(bool enable)
    {
        std_msgs::msg::Bool msg;
        msg.data = enable;
        listen_pub_->publish(msg);
    }

    // ── Navigate ──────────────────────────────────────────────────────────────
    bool navigateTo(double x, double y, double yaw_deg, bool check_detection = true)
    {
        if (!nav_client_->wait_for_action_server(5s)) {
            RCLCPP_ERROR(get_logger(), "Nav2 server not available");
            return false;
        }

        NavigateToPose::Goal goal;
        goal.pose.header.frame_id    = "map";
        goal.pose.header.stamp       = now();
        goal.pose.pose.position.x    = x;
        goal.pose.pose.position.y    = y;
        goal.pose.pose.position.z    = 0.0;
        double r                     = yaw_deg * M_PI / 180.0;
        goal.pose.pose.orientation.z = std::sin(r / 2.0);
        goal.pose.pose.orientation.w = std::cos(r / 2.0);

        { std::lock_guard<std::mutex> lock(nav_mutex_);
          nav_done_ = false; nav_success_ = false; }

        auto opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
        opts.goal_response_callback =
            [this](const GoalHandleNav::SharedPtr & handle) {
                std::lock_guard<std::mutex> lock(nav_mutex_);
                if (!handle) { nav_done_ = true; nav_cv_.notify_all(); return; }
                current_goal_handle_ = handle;
            };
        opts.feedback_callback = nullptr;
        opts.result_callback =
            [this](const GoalHandleNav::WrappedResult & result) {
                std::lock_guard<std::mutex> lock(nav_mutex_);
                nav_success_ = (result.code == rclcpp_action::ResultCode::SUCCEEDED);
                nav_done_    = true;
                nav_cv_.notify_all();
            };

        nav_client_->async_send_goal(goal, opts);

        while (rclcpp::ok()) {
            { std::unique_lock<std::mutex> lock(nav_mutex_);
              if (nav_cv_.wait_for(lock, 500ms, [this] { return nav_done_; })) break; }
            if (check_detection) {
                double cx = 0.0, cy = 0.0;
                if (detectObject(cx, cy)) { cancelNav(); stop(); return false; }
            }
        }
        return nav_success_;
    }

    // ── 360 scan ──────────────────────────────────────────────────────────────
    bool scan()
    {
        state_ = State::SCANNING;
        speak("Starting 360 scan");

        geometry_msgs::msg::Twist twist;
        twist.angular.z = SPIN_VEL;
        auto start = std::chrono::steady_clock::now();

        while (rclcpp::ok()) {
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= SPIN_DURATION) break;
            double cx = 0.0, cy = 0.0;
            if (detectObject(cx, cy)) { stop(); return false; }
            cmd_pub_->publish(twist);
            std::this_thread::sleep_for(100ms);
        }

        stop();
        speak("Scan complete");
        return true;
    }

    // ── Face object ───────────────────────────────────────────────────────────
    void faceObject()
    {
        int frame_w = 0;
        { std::lock_guard<std::mutex> lock(frame_mutex_);
          if (!latest_frame_.empty()) frame_w = latest_frame_.cols; }
        if (frame_w == 0) return;

        double center_x  = frame_w / 2.0;
        double tolerance = frame_w * 0.05;

        for (int i = 0; i < 100 && rclcpp::ok(); ++i) {
            double cx = 0.0, cy = 0.0;
            if (!detectObject(cx, cy)) break;
            double error = cx - center_x;
            if (std::abs(error) < tolerance) break;
            geometry_msgs::msg::Twist t;
            t.angular.z = (error > 0) ? -0.3 : 0.3;
            cmd_pub_->publish(t);
            std::this_thread::sleep_for(100ms);
        }
        stop();
    }

    // ── Wait for voice response ───────────────────────────────────────────────
    // No delay — mic turns on immediately after instruction is spoken
    std::string waitForVoiceResponse()
    {
        state_ = State::WAITING_RESPONSE;
        { std::lock_guard<std::mutex> lock(voice_mutex_);
          voice_received_ = false; voice_response_ = ""; }

        // Speak instruction then immediately enable mic
        speak("Say come back to return to start, or say find followed by object name to search for something new.");

        // Small sleep just to let espeak start before mic opens
        // so mic doesn't pick up the tail of the espeak audio
        std::this_thread::sleep_for(3s);

        // Enable mic
        setListening(true);

        // Wait up to 30s for response
        std::unique_lock<std::mutex> lock(voice_mutex_);
        bool got = voice_cv_.wait_for(lock, 30s, [this] { return voice_received_; });

        // Disable mic
        setListening(false);

        if (!got) return "timeout";
        return voice_response_;
    }

    // ── Return home ───────────────────────────────────────────────────────────
    void returnHome()
    {
        state_ = State::RETURNING;
        speak("Returning to start position");
        bool reached = navigateTo(HOME.x, HOME.y, HOME.yaw, false);
        speak(reached ? "Reached start position. Ready for new command."
                      : "Could not reach start position.");
        state_ = State::IDLE;
    }

    // ── Handle detection ──────────────────────────────────────────────────────
    void handleDetection()
    {
        state_ = State::DETECTED;
        stop();
        faceObject();

        double cx = 0.0, cy = 0.0;
        detectObject(cx, cy);

        double obj_x = 0.0, obj_y = 0.0;
        bool coords_ok = getObjectCoordinates(cx, cy, obj_x, obj_y);

        speak("Object detected. " + target_ + " has been found.");

        if (coords_ok) {
            char buf[128];
            snprintf(buf, sizeof(buf),
                "Object location in map: x is %.2f, y is %.2f", obj_x, obj_y);
            speak(std::string(buf));
            RCLCPP_INFO(get_logger(),
                "Object coordinates: x=%.4f y=%.4f", obj_x, obj_y);
        } else {
            speak("Could not calculate precise coordinates.");
        }

        std::string response = waitForVoiceResponse();

        if (response.find("come back") != std::string::npos ||
            response.find("go back")   != std::string::npos ||
            response.find("return")    != std::string::npos)
        { returnHome(); return; }

        auto pos = response.find("find ");
        if (pos != std::string::npos) {
            std::string new_target = response.substr(pos + 5);
            new_target.erase(new_target.find_last_not_of(" \n\r\t") + 1);
            if (!new_target.empty()) {
                target_ = new_target;
                speak("Command received. Finding " + target_);
                resetDetection();
                state_ = State::IDLE;
                std::thread(&SearchRobot::runMission, this).detach();
                return;
            }
        }

        speak(response == "timeout"
            ? "No response received. Returning to start."
            : "Command not understood. Returning to start.");
        returnHome();
    }

    // ── Main mission ──────────────────────────────────────────────────────────
    void runMission()
    {
        resetDetection();

        for (const auto & wp : WAYPOINTS) {
            resetDetection();
            state_ = State::MOVING;
            speak("Heading to waypoint");

            bool reached = navigateTo(wp.x, wp.y, wp.yaw, true);
            if (!reached) { handleDetection(); return; }

            speak("Reached waypoint");

            resetDetection();
            if (!scan()) { handleDetection(); return; }
        }

        speak("All waypoints visited. " + target_ + " was not found.");
        state_ = State::IDLE;
    }
};


int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SearchRobot>());
    rclcpp::shutdown();
    return 0;
}