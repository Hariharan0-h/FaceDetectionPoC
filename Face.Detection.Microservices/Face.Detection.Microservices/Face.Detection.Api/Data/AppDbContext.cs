using Face.Detection.Api.Entities;
using Microsoft.EntityFrameworkCore;

namespace Face.Detection.Api.Data
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
        {
        }

        public DbSet<Patients> Patients { get; set; }
    }
}
