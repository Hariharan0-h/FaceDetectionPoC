using Face.Detection.Api.Data;
using Face.Detection.Api.Entities;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Text.Json;

namespace Face.Detection.Api.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class PatientsController : ControllerBase
    {
        private readonly AppDbContext _context;
        private const double FACE_MATCH_THRESHOLD = 0.6; // Adjustable similarity threshold

        public PatientsController(AppDbContext context)
        {
            _context = context;
        }

        // GET: api/Patients
        [HttpGet]
        public async Task<ActionResult<IEnumerable<Patients>>> GetPatients()
        {
            return await _context.Patients.ToListAsync();
        }

        // GET: api/Patients/5
        [HttpGet("{id}")]
        public async Task<ActionResult<Patients>> GetPatients(int id)
        {
            var patients = await _context.Patients.FindAsync(id);
            if (patients == null)
            {
                return NotFound();
            }
            return patients;
        }

        // POST: api/Patients/FindByFace
        [HttpPost("FindByFace")]
        public async Task<ActionResult<PatientFaceMatchResult>> FindPatientByFaceEmbedding([FromBody] FaceEmbeddingRequest request)
        {
            try
            {
                if (request?.FaceEmbedding == null || !request.FaceEmbedding.Any())
                {
                    return BadRequest("Face embedding data is required");
                }

                var allPatients = await _context.Patients
                    .Where(p => !string.IsNullOrEmpty(p.FaceEmbedding))
                    .ToListAsync();

                PatientFaceMatchResult bestMatch = null;
                double bestSimilarity = 0;

                foreach (var patient in allPatients)
                {
                    try
                    {
                        var storedEmbedding = JsonSerializer.Deserialize<double[]>(patient.FaceEmbedding);
                        if (storedEmbedding != null && storedEmbedding.Length == request.FaceEmbedding.Length)
                        {
                            double similarity = CalculateCosineSimilarity(request.FaceEmbedding, storedEmbedding);

                            if (similarity > bestSimilarity && similarity >= FACE_MATCH_THRESHOLD)
                            {
                                bestSimilarity = similarity;
                                bestMatch = new PatientFaceMatchResult
                                {
                                    Patient = patient,
                                    Similarity = similarity,
                                    IsMatch = true
                                };
                            }
                        }
                    }
                    catch (JsonException)
                    {
                        // Skip patients with invalid face embedding data
                        continue;
                    }
                }

                if (bestMatch != null)
                {
                    return Ok(bestMatch);
                }

                return Ok(new PatientFaceMatchResult
                {
                    Patient = null,
                    Similarity = 0,
                    IsMatch = false,
                    Message = "No matching patient found"
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Error processing face recognition: {ex.Message}");
            }
        }

        // POST: api/Patients/VerifyIdentity
        [HttpPost("VerifyIdentity/{id}")]
        public async Task<ActionResult<FaceVerificationResult>> VerifyPatientIdentity(int id, [FromBody] FaceEmbeddingRequest request)
        {
            try
            {
                var patient = await _context.Patients.FindAsync(id);
                if (patient == null)
                {
                    return NotFound("Patient not found");
                }

                if (string.IsNullOrEmpty(patient.FaceEmbedding))
                {
                    return BadRequest("Patient has no stored face embedding for verification");
                }

                if (request?.FaceEmbedding == null || !request.FaceEmbedding.Any())
                {
                    return BadRequest("Face embedding data is required for verification");
                }

                var storedEmbedding = JsonSerializer.Deserialize<double[]>(patient.FaceEmbedding);
                if (storedEmbedding == null || storedEmbedding.Length != request.FaceEmbedding.Length)
                {
                    return BadRequest("Invalid stored face embedding data");
                }

                double similarity = CalculateCosineSimilarity(request.FaceEmbedding, storedEmbedding);
                bool isVerified = similarity >= FACE_MATCH_THRESHOLD;

                return Ok(new FaceVerificationResult
                {
                    PatientId = id,
                    PatientName = patient.Name,
                    IsVerified = isVerified,
                    Similarity = similarity,
                    Threshold = FACE_MATCH_THRESHOLD,
                    Message = isVerified ? "Identity verified successfully" : "Identity verification failed"
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Error during identity verification: {ex.Message}");
            }
        }

        // PUT: api/Patients/5
        [HttpPut("{id}")]
        public async Task<IActionResult> PutPatients(int id, Patients patients)
        {
            if (id != patients.Id)
            {
                return BadRequest();
            }

            _context.Entry(patients).State = EntityState.Modified;

            try
            {
                await _context.SaveChangesAsync();
            }
            catch (DbUpdateConcurrencyException)
            {
                if (!PatientsExists(id))
                {
                    return NotFound();
                }
                else
                {
                    throw;
                }
            }

            return NoContent();
        }

        // POST: api/Patients
        [HttpPost]
        public async Task<ActionResult<Patients>> PostPatients(Patients patients)
        {
            // Validate face embedding if provided
            if (!string.IsNullOrEmpty(patients.FaceEmbedding))
            {
                try
                {
                    var embedding = JsonSerializer.Deserialize<double[]>(patients.FaceEmbedding);
                    if (embedding == null || embedding.Length == 0)
                    {
                        return BadRequest("Invalid face embedding format");
                    }
                }
                catch (JsonException)
                {
                    return BadRequest("Face embedding must be a valid JSON array of numbers");
                }
            }

            _context.Patients.Add(patients);
            await _context.SaveChangesAsync();

            return CreatedAtAction("GetPatients", new { id = patients.Id }, patients);
        }

        // DELETE: api/Patients/5
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeletePatients(int id)
        {
            var patients = await _context.Patients.FindAsync(id);
            if (patients == null)
            {
                return NotFound();
            }

            _context.Patients.Remove(patients);
            await _context.SaveChangesAsync();

            return NoContent();
        }

        private bool PatientsExists(int id)
        {
            return _context.Patients.Any(e => e.Id == id);
        }

        // Helper method to calculate cosine similarity between two face embeddings
        private static double CalculateCosineSimilarity(double[] embedding1, double[] embedding2)
        {
            if (embedding1.Length != embedding2.Length)
                return 0;

            double dotProduct = 0;
            double norm1 = 0;
            double norm2 = 0;

            for (int i = 0; i < embedding1.Length; i++)
            {
                dotProduct += embedding1[i] * embedding2[i];
                norm1 += Math.Pow(embedding1[i], 2);
                norm2 += Math.Pow(embedding2[i], 2);
            }

            if (norm1 == 0 || norm2 == 0)
                return 0;

            return dotProduct / (Math.Sqrt(norm1) * Math.Sqrt(norm2));
        }

        // Alternative method using Euclidean distance (uncomment to use instead of cosine similarity)
        /*
        private static double CalculateEuclideanDistance(double[] embedding1, double[] embedding2)
        {
            if (embedding1.Length != embedding2.Length)
                return double.MaxValue;

            double sum = 0;
            for (int i = 0; i < embedding1.Length; i++)
            {
                sum += Math.Pow(embedding1[i] - embedding2[i], 2);
            }
            
            return Math.Sqrt(sum);
        }
        */
    }

    // DTO Classes for Face Recognition
    public class FaceEmbeddingRequest
    {
        public double[] FaceEmbedding { get; set; } = Array.Empty<double>();
    }

    public class PatientFaceMatchResult
    {
        public Patients? Patient { get; set; }
        public double Similarity { get; set; }
        public bool IsMatch { get; set; }
        public string Message { get; set; } = string.Empty;
    }

    public class FaceVerificationResult
    {
        public int PatientId { get; set; }
        public string PatientName { get; set; } = string.Empty;
        public bool IsVerified { get; set; }
        public double Similarity { get; set; }
        public double Threshold { get; set; }
        public string Message { get; set; } = string.Empty;
    }
}