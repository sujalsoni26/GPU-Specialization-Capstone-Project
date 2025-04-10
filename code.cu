#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>

#define BLOCK_SIZE 256

// Simulation parameters
const int N = 16384;              // Number of bodies
const float G = 6.67430e-11f;     // Gravitational constant
const float SOFTENING = 1.0f;     // Softening parameter to avoid singularities
float timeStep = 0.01f;           // Simulation time step
bool paused = false;              // Pause simulation flag

// Window dimensions
const int WIDTH = 1280;
const int HEIGHT = 720;

// Particle data structure
struct Particle {
    float x, y, z;        // Position
    float vx, vy, vz;     // Velocity
    float mass;           // Mass
};

// Host and device arrays
Particle* h_particles = nullptr;
Particle* d_particles = nullptr;
float4* d_positions = nullptr;

// GPU acceleration particle update kernel
__global__ void updateParticles(Particle* particles, float4* positions, float dt, int n, float G, float softening) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        float px = particles[i].x;
        float py = particles[i].y;
        float pz = particles[i].z;
        float pmass = particles[i].mass;
        
        // Calculate forces from all other particles
        for (int j = 0; j < n; j++) {
            if (i != j) {
                float dx = particles[j].x - px;
                float dy = particles[j].y - py;
                float dz = particles[j].z - pz;
                
                float distSqr = dx*dx + dy*dy + dz*dz + softening*softening;
                float distSixth = distSqr * distSqr * distSqr;
                float invDist = 1.0f / sqrtf(distSixth);
                
                float force = G * pmass * particles[j].mass * invDist;
                
                fx += force * dx;
                fy += force * dy;
                fz += force * dz;
            }
        }
        
        // Update velocity
        particles[i].vx += fx * dt / pmass;
        particles[i].vy += fy * dt / pmass;
        particles[i].vz += fz * dt / pmass;
        
        // Update position
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
        
        // Update positions array for rendering
        positions[i] = make_float4(particles[i].x, particles[i].y, particles[i].z, pmass);
    }
}

// Initialize particles with random positions and velocities
void initParticles() {
    h_particles = new Particle[N];
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> posDist(-1000.0f, 1000.0f);
    std::uniform_real_distribution<float> velDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> massDist(1.0f, 10000.0f);
    
    // Create a galactic disk formation
    for (int i = 0; i < N; i++) {
        float radius = std::abs(posDist(gen)) * 0.5f;
        float angle = posDist(gen) * 0.02f;
        
        h_particles[i].x = radius * cos(angle);
        h_particles[i].y = posDist(gen) * 0.1f;
        h_particles[i].z = radius * sin(angle);
        
        // Orbital velocity for roughly stable orbits
        float orbitalSpeed = sqrt(G * 1e15 / radius) * 0.5f;
        h_particles[i].vx = -sin(angle) * orbitalSpeed;
        h_particles[i].vy = velDist(gen) * 0.1f;
        h_particles[i].vz = cos(angle) * orbitalSpeed;
        
        // Mass of particle
        h_particles[i].mass = massDist(gen);
    }
    
    // Add a massive central body
    h_particles[0].x = 0.0f;
    h_particles[0].y = 0.0f;
    h_particles[0].z = 0.0f;
    h_particles[0].vx = 0.0f;
    h_particles[0].vy = 0.0f;
    h_particles[0].vz = 0.0f;
    h_particles[0].mass = 1e15f;  // Massive central body
    
    // Allocate device memory
    cudaMalloc((void**)&d_particles, N * sizeof(Particle));
    cudaMalloc((void**)&d_positions, N * sizeof(float4));
    
    // Copy particle data to device
    cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice);
}

// OpenGL setup function
GLuint setupOpenGL() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(-1);
    }
    
    // Create window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "GPU N-Body Simulation", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }
    
    glfwMakeContextCurrent(window);
    
    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        exit(-1);
    }
    
    // Shader sources
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec4 position;
        uniform mat4 projection;
        uniform mat4 view;
        
        out float pointSize;
        out float brightness;
        
        void main() {
            gl_Position = projection * view * vec4(position.xyz, 1.0);
            
            // Calculate point size based on mass and distance
            float mass = position.w;
            float dist = length(position.xyz);
            pointSize = 0.5 + sqrt(mass) * 0.005f;
            brightness = min(1.0, 0.2 + mass * 0.00005f);
            
            // Extra size for the central body
            if (mass > 1e14) {
                pointSize = 10.0;
                brightness = 1.0;
            }
            
            gl_PointSize = pointSize;
        }
    )";
    
    const char* fragmentShaderSource = R"(
        #version 330 core
        in float pointSize;
        in float brightness;
        out vec4 FragColor;
        
        void main() {
            // Calculate distance from center of point
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            
            // Discard pixels outside of circle
            if (dist > 0.5) {
                discard;
            }
            
            // Generate star color (white-blue gradient based on mass)
            vec3 color = mix(vec3(0.6, 0.8, 1.0), vec3(1.0), brightness);
            
            // Smooth edges and apply glow effect
            float alpha = 1.0 - smoothstep(0.45, 0.5, dist);
            
            FragColor = vec4(color, alpha);
        }
    )";
    
    // Create shaders
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    
    // Create shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    // Check shader compilation
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    // Delete shaders as they're linked into the program and no longer needed
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return shaderProgram;
}

// Keyboard callback function
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        paused = !paused;
    }

    if (key == GLFW_KEY_UP && action == GLFW_PRESS) {
        timeStep *= 1.5f;
        std::cout << "Time step: " << timeStep << std::endl;
    }

    if (key == GLFW_KEY_DOWN && action == GLFW_PRESS) {
        timeStep /= 1.5f;
        std::cout << "Time step: " << timeStep << std::endl;
    }
}

int main() {
    // Initialize particles
    initParticles();

    // Setup OpenGL
    GLuint shaderProgram = setupOpenGL();
    GLFWwindow* window = glfwGetCurrentContext();
    
    // Set key callback
    glfwSetKeyCallback(window, key_callback);
    
    // Create VBO and VAO for rendering
    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Register VBO as CUDA resource
    cudaGraphicsResource* cuda_vbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsRegisterFlagsWriteDiscard);
    
    // Enable alpha blending for point sprites
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    
    // Camera setup
    glm::vec3 cameraPos = glm::vec3(0.0f, 500.0f, 1500.0f);
    glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    
    float lastFrame = 0.0f;
    float rotationAngle = 0.0f;
    float cameraDistance = 1500.0f;
    float cameraHeight = 500.0f;

    // Main loop
    std::cout << "Controls:" << std::endl;
    std::cout << "ESC: Exit" << std::endl;
    std::cout << "SPACE: Pause/Resume simulation" << std::endl;
    std::cout << "UP/DOWN: Increase/Decrease simulation speed" << std::endl;
    
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        float currentFrame = glfwGetTime();
        float deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        
        // Process input
        glfwPollEvents();
        
        // Rotate camera
        rotationAngle += deltaTime * 0.1f;
        cameraPos.x = sin(rotationAngle) * cameraDistance;
        cameraPos.z = cos(rotationAngle) * cameraDistance;
        cameraPos.y = cameraHeight;
        
        // Update particles if not paused
        if (!paused) {
            // Launch CUDA kernel to update particles
            int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            updateParticles<<<numBlocks, BLOCK_SIZE>>>(d_particles, d_positions, timeStep, N, G, SOFTENING);
            
            // Map OpenGL buffer object for writing from CUDA
            float4* dptr;
            size_t size;
            cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
            cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, cuda_vbo_resource);
            
            // Copy updated positions to mapped buffer
            cudaMemcpy(dptr, d_positions, N * sizeof(float4), cudaMemcpyDeviceToDevice);
            
            // Unmap buffer
            cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
        }
        
        // Render particles
        glClearColor(0.0f, 0.0f, 0.05f, 1.0f); // Dark blue background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Update camera and projection matrices
        glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)WIDTH / (float)HEIGHT, 0.1f, 10000.0f);
        
        // Use shader program
        glUseProgram(shaderProgram);
        
        // Set uniform values
        GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
        
        // Render particles
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, N);
        
        // Swap buffers
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    
    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    cudaFree(d_particles);
    cudaFree(d_positions);
    delete[] h_particles;
    
    glfwTerminate();
    return 0;
}
